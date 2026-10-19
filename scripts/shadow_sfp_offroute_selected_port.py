#!/usr/bin/env python3
"""Frozen observation-only board/crop replay on actor-prefix development views."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import cv2
cv2.setNumThreads(1)
import numpy as np
import torch

from audit_sc_port_targets import pose_matrix
from build_two_family_port_projection_labels import sfp_ports
from evaluate_sc_native_triangulation import triangulate, reprojection_residual_px
from profile_sfp_coarse_crop_latency import read
from train_sfp_board_reference import BoardReferenceLocal, SIZE
from train_sfp_observed_rail_crop import crop_native
from train_target_conditioned_port_locator import ConditionalLocator
from train_two_family_port_locator import CAMERAS, LOOKUP


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--board',type=Path,required=True)
    p.add_argument('--crop',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stride-seconds',type=float,default=1.)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve prior shadow replay')
    if a.stride_seconds<.5:p.error('Use a bounded stride')
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    board_state=torch.load(a.board,map_location='cpu',weights_only=False)
    board=BoardReferenceLocal(board_state['init']).cuda().eval()
    board.load_state_dict(board_state['model'])
    crop=ConditionalLocator(None).cuda().eval()
    crop.load_state_dict(torch.load(a.crop,map_location='cpu',weights_only=False)['model'])
    optics_json=json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())
    optics={c:np.asarray(optics_json['tcp_to_optical'][c],float) for c in CAMERAS}
    port_cal=json.loads(Path('configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json').read_text())
    source_manifest=json.loads((a.labels.parent/'manifest.json').read_text())
    audits={8100000+i:episode['audit'] for i,episode in enumerate(source_manifest['episodes'])}
    ports={}
    for episode,audit in audits.items():
        row=source_manifest['episodes'][episode-8100000]
        ports[episode]={(card,port):m for card,port,m in sfp_ports(
            {'trial_yaml':str(Path(audit)/'eval_config.yaml'),'trial_id':row['trial_id']},port_cal)}
    selected=[];last=defaultdict(lambda:-1e9)
    for row in map(json.loads,a.labels.open()):
        if not row.get('actor_prefix_context_only'):continue
        episode=int(row['episode_index']);stamp=float(row['timestamp_s'])
        if episode not in audits or stamp-last[episode]<a.stride_seconds:continue
        selected.append(row);last[episode]=stamp
    if len(selected)<15 or len({r['episode_index'] for r in selected})!=3:
        p.error('Expected three actor-prefix episodes and at least fifteen fixed-stride moments')
    predictions=[];times=[]
    for row in selected:
        t0=time.perf_counter()
        ep=int(row['episode_index']);card=int(row['selected_card']);port=int(row['selected_port'])
        requested=f'sfp_card_{card}_port_{port}'
        images={c:cv2.imread(row['images'][c],cv2.IMREAD_COLOR) for c in CAMERAS}
        if any(x is None for x in images.values()):raise FileNotFoundError(row['images'])
        board_rgb=np.stack([cv2.cvtColor(cv2.resize(images[c],SIZE,interpolation=cv2.INTER_AREA),
                                        cv2.COLOR_BGR2RGB).transpose(2,0,1) for c in CAMERAS])
        board_input=torch.from_numpy(board_rgb.copy()).cuda().float()[None]/255.
        with torch.inference_mode():ends,_,_,_=board(board_input)
        centers={}
        for ci,cam in enumerate(CAMERAS):
            endpoints=ends[0,ci].float().cpu().numpy()*[1151,1023]
            centers[cam]=endpoints[0]+card/4*(endpoints[1]-endpoints[0])
        patches=np.stack([crop_native(images[c],centers[c]) for c in CAMERAS])
        crop_input=torch.from_numpy(patches.copy()).cuda().float().permute(0,3,1,2)/255.
        meta=np.zeros((3,5),np.float32);meta[:,1]=1;meta[np.arange(3),np.arange(3)+2]=1
        with torch.inference_mode():xy,flag,_=crop(crop_input,torch.from_numpy(meta).cuda(),
            torch.full((3,),LOOKUP[requested],device='cuda',dtype=torch.long))
        xy=xy.float().cpu().numpy();prob=flag.sigmoid().float().cpu().numpy()
        state=np.asarray(row['state'],float)
        optical={c:pose_matrix(state)@optics[c] for c in CAMERAS}
        pixels={c:centers[c]+(xy[ci]-.5)*480 for ci,c in enumerate(CAMERAS) if prob[ci]>=.5}
        estimated=None;reprojection=None;accepted=False
        if len(pixels)==3:
            try:
                estimated=triangulate(pixels,optical)
                reprojection=float(reprojection_residual_px(estimated,pixels,optical))
                accepted=bool(np.isfinite(estimated).all() and reprojection<=5)
            except np.linalg.LinAlgError:pass
        torch.cuda.synchronize();latency=(time.perf_counter()-t0)*1000;times.append(latency)
        # Geometry enters only after RGB/task/state prediction and timing are frozen.
        truth=ports[ep][card,port]
        relative=truth[:3,:3].T@(state[:3]-truth[:3,3])
        nearest=None;error=None
        if estimated is not None:
            nearest=min(ports[ep],key=lambda key:np.linalg.norm(estimated-ports[ep][key][:3,3]))
            error=float(np.linalg.norm(estimated-truth[:3,3])*1000)
        predictions.append(dict(episode=ep,time_s=float(row['timestamp_s']),requested=[card,port],
            accepted=accepted,probabilities={c:float(prob[i]) for i,c in enumerate(CAMERAS)},
            predicted_base_m=estimated.tolist() if estimated is not None else None,
            reprojection_px=reprojection,nearest_identity_label_only=list(nearest) if nearest else None,
            requested_error_mm_label_only=error,axial_mm_label_only=float(relative[2]*1000),
            lateral_mm_label_only=float(np.linalg.norm(relative[:2])*1000),latency_ms=latency))
    a.output.mkdir(parents=True)
    with (a.output/'predictions.jsonl').open('w') as f:
        for row in predictions:f.write(json.dumps(row,separators=(',',':'))+'\n')
    grouped=defaultdict(list)
    for row in predictions:grouped[row['episode']].append(row)
    episodes=[];all_accepted=[];wrong=0
    for episode,rows in grouped.items():
        rows.sort(key=lambda x:x['time_s'])
        band=next((x['time_s'] for x in rows if -100<=x['axial_mm_label_only']<=-70),None)
        before=[x for x in rows if band is None or x['time_s']<band]
        accepts=[x for x in before if x['accepted']]
        all_accepted.extend(accepts)
        mistakes=sum(x['nearest_identity_label_only']!=x['requested'] for x in accepts)
        wrong+=mistakes
        episodes.append(dict(episode=episode,sampled=len(rows),first_safe_axial_band_s=band,
            decisions_before_band=len(before),accepted_before_band=len(accepts),
            correct_before_band=sum(x['nearest_identity_label_only']==x['requested'] for x in accepts),
            wrong_before_band=mistakes,first_correct_s=next((x['time_s'] for x in accepts
                if x['nearest_identity_label_only']==x['requested']),None)))
    errors=[x['requested_error_mm_label_only'] for x in all_accepted]
    good=sum(x['correct_before_band']>0 for x in episodes)
    p50,p95,p99=np.percentile(times,[50,95,99])
    summary=dict(schema='aic_sfp_offroute_selected_port_shadow/v1',
        source=str(a.labels),board=str(a.board),crop=str(a.crop),sample_stride_s=a.stride_seconds,
        sampled=len(predictions),episodes=episodes,episodes_correct_before_band=good,
        wrong_accepts_before_band=wrong,accepted_3d_error_p95_mm=float(np.percentile(errors,95)) if errors else None,
        complete_saved_image_decode_and_model_latency_p50_p95_p99_ms=[float(p50),float(p95),float(p99)],
        gate_passed=bool(good>=2 and wrong==0 and p95<300),
        runtime_true_geometry_input=False,geometry_label_only_after_prediction=True,
        autonomous_handoff_enabled=False,final_scenes_opened=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
