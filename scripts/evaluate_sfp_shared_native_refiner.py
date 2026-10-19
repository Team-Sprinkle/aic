#!/usr/bin/env python3
"""Match stronger SFP coarse baseline with/without the retrained native refiner.

Only RGB, public request, fixed calibration and measured TCP enter prediction.
GT port poses/other identities are read afterward for scoring. Fixed grouped
validation recordings may overlap older component development data; promotion
still needs new live reset groups, never the reserved final scenes.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from sfp_selected_port_observation import SelectedSfpPort


def statistics(v):
    return dict(n=len(v),p50=float(np.median(v)) if v else None,p95=float(np.percentile(v,95)) if v else None)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--board',type=Path,required=True);p.add_argument('--coarse-crop',type=Path,required=True)
    p.add_argument('--refiner',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames-per-episode',type=int,default=16)
    p.add_argument('--pose-mount-priors',type=Path)
    p.add_argument('--minimum-views',type=int,choices=(2,3),default=3)
    p.add_argument('--device',default='cuda')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(2);torch.set_num_threads(6)
    episodes=defaultdict(list)
    for l in a.labels.open():
        r=json.loads(l)
        if r['family']=='sfp' and r['split']=='validation' and max(r['image_sim_ns'].values())-min(r['image_sim_ns'].values())<=1_000_000:
            episodes[r['episode']].append(r)
    rows=[]
    for ep,rr in sorted(episodes.items()):
        rr.sort(key=lambda r:r['time_s']);rows.extend(rr[i] for i in np.linspace(0,len(rr)-1,min(len(rr),a.frames_per_episode),dtype=int))
    # Freeze selected images/requests before inference; no GT distance or
    # model outputs determine which examples enter the comparison.
    with (a.output/'manifest.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps({k:r[k] for k in ('episode','scene_group','time_s','selected_identity','images','state','card_count')})+'\n')
    observer=SelectedSfpPort(a.board,a.coarse_crop,a.device,native_refiner_checkpoint=a.refiner,native_pose_mount_priors=a.pose_mount_priors,minimum_views=a.minimum_views)
    fixture_mount=observer.fixture_mount
    native=observer.native_refiner;observer.warmup();records=[]
    with (a.output/'predictions.jsonl').open('w') as writer:
        for start in range(0,len(rows),16):
            batch=rows[start:start+16];items=[]
            for r in batch:
                parts=r['selected_identity'].split('_');card=int(parts[2]);port=int(parts[-1])
                images={c:cv2.imread(r['images'][c]) for c in ('center','left','right')}
                items.append((images,np.asarray(r['state']),card,port))
            for variant,head in [('strong_coarse',None),('strong_coarse_native_refined',native)]:
                observer.native_refiner=head;observer.fixture_mount=fixture_mount if head is not None else None
                pred=observer.predict_batch(items)
                for r,result in zip(batch,pred):
                    out=dict(episode=r['episode'],scene_group=r['scene_group'],time_s=r['time_s'],requested=r['selected_identity'],cards=r['card_count'],variant=variant,
                        accepted=result['accepted'],reprojection_px=result['reprojection_px'],view_probabilities=result['view_probabilities'],coarse_pixels=result['coarse_pixels'],refined_pixels=result['refined_pixels'],geometry_prediction_input=False)
                    if result['accepted']:
                        target=next(t for t in r['targets'] if t['identity']==r['selected_identity']);point=np.asarray(result['predicted_base_m']);error=Rotation.from_quat(target['orientation_base_xyzw']).inv().apply(point-target['position_base_m'])*1000
                        nearest=min(r['targets'],key=lambda t:np.linalg.norm(point-t['position_base_m']))
                        out.update(predicted_position_base_m=point.tolist(),xyz_error_mm=float(np.linalg.norm(error)),lateral_error_mm=float(np.linalg.norm(error[:2])),axial_error_mm=float(abs(error[2])),nearest_GT_identity_for_scoring=nearest['identity'],wrong_identity=nearest['identity']!=r['selected_identity'])
                        if result.get('predicted_orientation_base_xyzw') is not None:
                            out['predicted_orientation_base_xyzw']=result['predicted_orientation_base_xyzw']
                            out['orientation_error_deg']=float(np.degrees((Rotation.from_quat(target['orientation_base_xyzw']).inv()*Rotation.from_quat(out['predicted_orientation_base_xyzw'])).magnitude()))
                    records.append(out);writer.write(json.dumps(out)+'\n')
            print(json.dumps(dict(rows_completed=min(start+16,len(rows)),total_rows=len(rows))),flush=True)
    report={}
    for variant in ('strong_coarse','strong_coarse_native_refined'):
        rr=[r for r in records if r['variant']==variant];accepted=[r for r in rr if r['accepted']]
        report[variant]=dict(rows=len(rr),episodes=len({r['episode'] for r in rr}),accepted=len(accepted),episodes_with_acceptance=len({r['episode'] for r in accepted}),wrong_accepts=sum(r['wrong_identity'] for r in accepted),
            errors={key:statistics([r[key] for r in accepted]) for key in ('xyz_error_mm','lateral_error_mm','axial_error_mm')},
            orientation_error_deg=statistics([r['orientation_error_deg'] for r in accepted if 'orientation_error_deg' in r]),
            by_card_count={str(n):dict(rows=sum(r['cards']==n for r in rr),accepted=sum(r['cards']==n for r in accepted),wrong=sum(r['cards']==n and r['wrong_identity'] for r in accepted),xyz_mm=statistics([r['xyz_error_mm'] for r in accepted if r['cards']==n])) for n in range(1,6)})
    # Batch1/preloaded-image timing includes resize, models and result transfer.
    # Camera capture, ROS/controller and insertion actor are explicitly excluded.
    timings={}
    for variant,head in [('strong_coarse',None),('strong_coarse_native_refined',native)]:
        observer.native_refiner=head;observer.fixture_mount=fixture_mount if head is not None else None;times=[]
        for item in items[:5]:observer.predict_batch([item])
        for i in range(60):
            if observer.device.type=='cuda':torch.cuda.synchronize(observer.device)
            t=time.perf_counter();observer.predict_batch([items[i%len(items)]])
            if observer.device.type=='cuda':torch.cuda.synchronize(observer.device)
            times.append((time.perf_counter()-t)*1000)
        timings[variant]=dict(p50_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),p99_ms=float(np.percentile(times,99)),scope='batch1 preloaded-image perception path; excludes camera/ROS/controller/actor')
    summary=dict(schema='SFP_strong_coarse_shared_refiner_comparison/v1',results=report,latency=timings,
        device=str(observer.device),minimum_views=a.minimum_views,
        selected_rows_without_GT_filter=True,physical_visibility_certified=False,full_orientation_not_evaluated=fixture_mount is None,
        component_overlap_with_older_training_not_excluded=True,fresh_live_promotion_required=True,final_scenes_opened=False,promoted=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
