#!/usr/bin/env python3
"""Time RGB coarse locator + SC identity head + native crop refiner in memory.

This is a component benchmark, excluding ROS transport, actor and controller.
"""

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from train_sc_spatial_identity_probe import CAMERAS, SelectedHead
from train_sc_native_keypoints import PretrainedKeypointRefiner
from train_two_family_port_locator import Locator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows',type=Path,required=True)
    p.add_argument('--locator',type=Path,required=True)
    p.add_argument('--identity',type=Path,required=True)
    p.add_argument('--refiner',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--iterations',type=int,default=100)
    a=p.parse_args();cv2.setNumThreads(1);torch.set_num_threads(4)
    rows=[json.loads(x) for x in a.rows.open()]
    selected=[next(r for r in rows if r['camera']==c and r['projected_inframe']) for c in CAMERAS]
    native=[cv2.imread(r['image']) for r in selected]
    if any(x is None for x in native):raise ValueError('Missing RGB')
    request=int(selected[0]['target_port'])
    locator=Locator(None).to(a.device).eval()
    locator.load_state_dict(torch.load(a.locator,map_location='cpu',weights_only=False)['model'])
    saved=torch.load(a.identity,map_location='cpu',weights_only=False)
    model=SelectedHead(True,256,12).to(a.device).eval();model.load_state_dict(saved['model'])
    refiner=PretrainedKeypointRefiner(None,False,point_count=1).to(a.device).eval()
    refiner.load_state_dict(torch.load(a.refiner,map_location='cpu',weights_only=False)['model'])
    state=torch.tensor([r['measured_tcp_base_pose'] for r in selected],device=a.device,dtype=torch.float32)
    state=(state-saved['state_mean'].to(a.device))/saved['state_std'].to(a.device)
    camera=torch.eye(3,device=a.device)
    request_oh=torch.zeros((3,2),device=a.device);request_oh[:,request]=1
    condition=torch.cat((camera,state,request_oh),1)
    meta=torch.cat((torch.tensor([[1.,0.]],device=a.device).expand(3,-1),camera),1)

    @torch.inference_mode()
    def iteration():
        small=np.stack([cv2.cvtColor(cv2.resize(im,(288,256),interpolation=cv2.INTER_AREA),
                                        cv2.COLOR_BGR2RGB) for im in native])
        rgb=torch.from_numpy(small).permute(0,3,1,2).to(a.device).float()/255
        coords,_=locator(rgb,meta)
        x=(rgb-locator.mean)/locator.std
        x=locator.backbone.maxpool(locator.backbone.relu(locator.backbone.bn1(locator.backbone.conv1(x))))
        x=locator.backbone.layer3(locator.backbone.layer2(locator.backbone.layer1(x)))
        score=model(x,condition).sigmoid()
        points=(coords[:,request].float().cpu().numpy()*[1151,1023]).astype(np.float32)
        crops=[]
        for im,point in zip(native,points):
            left,top=np.round(point-80).astype(int)
            crop=cv2.warpAffine(im,np.array([[1,0,-left],[0,1,-top]],np.float32),
                                (160,160),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            crops.append(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))
        tensor=torch.from_numpy(np.stack(crops)).permute(0,3,1,2).to(a.device).float()/255
        with torch.autocast('cuda',dtype=torch.bfloat16): refined=refiner(tensor)
        torch.cuda.synchronize(a.device)
        return score,refined

    for _ in range(20):iteration()
    values=[]
    for _ in range(a.iterations):
        start=time.perf_counter();iteration();values.append((time.perf_counter()-start)*1000)
    result={'schema':'sc_selected_identity_component_latency/v1','iterations':len(values),
        'p50_ms':float(np.quantile(values,.5)),'p95_ms':float(np.quantile(values,.95)),
        'p99_ms':float(np.quantile(values,.99)),'max_ms':float(max(values)),
        'component_only_excludes':['ROS image delivery','transport actor','insertion actor',
                                   'camera calibration/triangulation','controller command'],
        'three_native_rgb_preloaded':True,'locator_runs_twice_upper_bound_for_shared_backbone':True,
        'sample_images':[r['image'] for r in selected],'requested_port':request,
        'final_scenes_opened':False}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
