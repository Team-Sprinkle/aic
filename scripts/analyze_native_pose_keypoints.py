#!/usr/bin/env python3
"""Compare unconstrained keypoint pose with a declared flat-fixture fit.

GT is used only for training-derived fixed mount calibration and posthoc error
metrics. Predicted landmarks and public robot/camera poses determine each fit.
This is conditional crop accuracy, not autonomous acquisition or visibility.
"""
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
from audit_sc_port_targets import pose_matrix
from evaluate_sc_native_triangulation import FOCAL
from native_keypoint_pose_geometry import fit_fixed_normal_pose
from train_native_requested_pose_keypoints import reference_points


def stats(v):
    return dict(n=len(v),p50=float(np.median(v)) if v else None,p95=float(np.percentile(v,95)) if v else None)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    mount_values=defaultdict(list);mount_groups=defaultdict(list)
    with (a.run/'manifest.jsonl').open() as f:
        for l in f:
            r=json.loads(l)
            if r['split']!='train':continue
            rotation=Rotation.from_quat(r['target_orientation_base_xyzw']).as_matrix();yaw=np.arctan2(rotation[1,0],rotation[0,0])
            mount_values[r['family']].append(Rotation.from_euler('z',-yaw).as_matrix()@rotation);mount_groups[r['family']].append(r['scene_group'])
    mounts={}
    for family,v in mount_values.items():
        counts=Counter(mount_groups[family]);mounts[family]=Rotation.from_matrix(np.stack(v)).mean(weights=[1/counts[g] for g in mount_groups[family]]).as_matrix()
    (a.output/'training_only_fixture_mount_priors.json').write_text(json.dumps({f:r.tolist() for f,r in mounts.items()},indent=2)+'\n')
    optics={c:np.asarray(v) for c,v in json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())['tcp_to_optical'].items()}
    groups=defaultdict(dict)
    with (a.run/'validation_predictions.jsonl').open() as f:
        for l in f:
            r=json.loads(l)
            if r['selected']:groups[(r['family'],r['episode'],r['time_s'])][r['camera']]=r
    errors=defaultdict(lambda:defaultdict(list));rejected=Counter();latency=[];records=[]
    for key,views in groups.items():
        if len(views)!=3:rejected[key[0]+'/missing_three_labelled_views']+=1;continue
        r=next(iter(views.values()));stamps=[int(Path(v['source_image']).stem.split('_')[-2]) for v in views.values()]
        if max(stamps)-min(stamps)>1_000_000:rejected[key[0]+'/asynchronous']+=1;continue
        cameras={c:pose_matrix(r['state'])@optics[c] for c in views}
        tick=time.perf_counter()
        try:fit=fit_fixed_normal_pose({c:np.asarray(v['predicted_keypoints_xy']) for c,v in views.items()},cameras,reference_points(r['family']),mounts[r['family']],FOCAL)
        except (ValueError,np.linalg.LinAlgError):rejected[key[0]+'/fit_failed']+=1;continue
        latency.append((time.perf_counter()-tick)*1000)
        rotation=Rotation.from_quat(r['target_orientation_base_xyzw']);delta=rotation.inv().apply(fit['position_base_m']-r['target_position_base_m'])*1000
        angular=float(np.degrees((rotation.inv()*Rotation.from_matrix(fit['rotation_base'])).magnitude()))
        record=dict(family=r['family'],episode=r['episode'],time_s=r['time_s'],xyz_mm=float(np.linalg.norm(delta)),lateral_mm=float(np.linalg.norm(delta[:2])),orientation_deg=angular,
            rms_reprojection_px=fit['rms_reprojection_px'],max_reprojection_px=fit['max_reprojection_px'],fit_converged=fit['converged'],geometry_prediction_input=False,
            all_centres_in_crop_margin=all(v['inside_margin'] for v in views.values()))
        records.append(record)
        buckets=[r['family']+'/all_three_labelled_views']
        if record['all_centres_in_crop_margin']:buckets.append(r['family']+'/inside_crop_centres')
        if fit['all_depths_positive'] and fit['max_reprojection_px']<=2:buckets.append(r['family']+'/predicted_reprojection_le2px')
        for b in buckets:
            for metric in ('xyz_mm','lateral_mm','orientation_deg'):errors[b][metric].append(record[metric])
    with (a.output/'predictions.jsonl').open('w') as f:
        for r in records:f.write(json.dumps(r)+'\n')
    report=dict(schema='fixed_normal_native_keypoint_pose_analysis/v1',metrics={k:{m:stats(v) for m,v in metrics.items()} for k,metrics in errors.items()},
        rejected=dict(rejected),CPU_fit_latency_ms=stats(latency),fixture_priors_from_training_only=True,
        conditional_crop_accuracy_only=True,physical_visibility_not_certified=True,
        arbitrary_board_pitch_roll_not_supported=True,final_scenes_opened=False,promoted=False)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
