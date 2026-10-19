#!/usr/bin/env python3
"""Measure full observation-only SFP board-to-port inference on native RGB."""
import argparse
import json
from pathlib import Path
import time

import cv2
cv2.setNumThreads(1)
import numpy as np
import torch

from audit_sc_port_targets import pose_matrix
from evaluate_sc_native_triangulation import triangulate, reprojection_residual_px
from train_sfp_board_reference import BoardReferenceLocal, SIZE
from train_sfp_observed_rail_crop import crop_native
from train_target_conditioned_port_locator import ConditionalLocator
from train_two_family_port_locator import CAMERAS, LOOKUP


def read(path):
    return [json.loads(s) for s in (path/'rows.jsonl').open()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--proposals',type=Path,required=True)
    p.add_argument('--board',type=Path,required=True)
    p.add_argument('--crop',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=60)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing profile')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    board_state=torch.load(a.board,map_location='cpu',weights_only=False)
    board=BoardReferenceLocal(board_state['init']).cuda().eval()
    board.load_state_dict(board_state['model'])
    crop=ConditionalLocator(None).cuda().eval()
    crop.load_state_dict(torch.load(a.crop,map_location='cpu',weights_only=False)['model'])
    labels={(x['episode'],x['time_s']):x for x in read(a.labels)}
    examples=[]
    for p_row in read(a.proposals):
        if len(examples)>=20:break
        r=labels[(p_row['episode'],p_row['time_s'])]
        images={cam:cv2.imread(r['images'][cam],cv2.IMREAD_COLOR) for cam in CAMERAS}
        if any(x is None for x in images.values()):raise FileNotFoundError(r['images'])
        examples.append((images,np.asarray(r['state'],float),p_row['requested']))
    if len(examples)<10:p.error('Need at least ten native-RGB examples')
    cal=json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())
    optics={cam:np.asarray(cal['tcp_to_optical'][cam],float) for cam in CAMERAS}
    def infer(example):
        images,state,requested=example
        board_rgb=np.stack([cv2.cvtColor(cv2.resize(images[c],SIZE,interpolation=cv2.INTER_AREA),
                                        cv2.COLOR_BGR2RGB).transpose(2,0,1) for c in CAMERAS])
        board_input=torch.from_numpy(board_rgb.copy()).cuda().float()[None]/255.
        with torch.inference_mode():ends,flags,occupied,_=board(board_input)
        card=int(requested.split('_')[2])
        centers={}
        for ci,cam in enumerate(CAMERAS):
            ends_px=ends[0,ci].float().cpu().numpy()*[1151,1023]
            centers[cam]=ends_px[0]+card/4*(ends_px[1]-ends_px[0])
        local=np.stack([crop_native(images[c],centers[c]) for c in CAMERAS])
        local_input=torch.from_numpy(local.copy()).cuda().float().permute(0,3,1,2)/255.
        meta=np.zeros((3,5),np.float32);meta[:,1]=1;meta[np.arange(3),np.arange(3)+2]=1
        with torch.inference_mode():xy,flag,_=crop(local_input,torch.from_numpy(meta).cuda(),
            torch.full((3,),LOOKUP[requested],device='cuda',dtype=torch.long))
        xy=xy.float().cpu().numpy();flag=flag.sigmoid().float().cpu().numpy()
        points={cam:centers[cam]+(xy[ci]-.5)*480 for ci,cam in enumerate(CAMERAS)
                if flag[ci]>=.5}
        if len(points)==3:
            cameras={c:pose_matrix(state)@optics[c] for c in CAMERAS}
            try:
                position=triangulate(points,cameras)
                _=reprojection_residual_px(position,points,cameras)
            except np.linalg.LinAlgError:pass
        return float(occupied.sigmoid()[0,card].item())
    for i in range(10):infer(examples[i%len(examples)])
    torch.cuda.synchronize();times=[]
    for i in range(a.samples):
        started=time.perf_counter();infer(examples[i%len(examples)])
        torch.cuda.synchronize();times.append((time.perf_counter()-started)*1000)
    result=dict(schema='aic_sfp_coarse_crop_latency/v1',samples=len(times),
                original_images_preloaded=True,includes_preprocessing=True,
                includes_board_and_crop_networks=True,includes_triangulation=True,
                p50_p95_p99_ms=np.percentile(times,[50,95,99]).tolist(),
                max_ms=float(max(times)),requested_ports_from_public_task_code=True,
                true_port_geometry_runtime_input=False,final_scenes_opened=False,
                board=str(a.board),crop=str(a.crop))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
