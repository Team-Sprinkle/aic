#!/usr/bin/env python3
"""Append targeted native RGB features to an immutable family cache copy."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from train_two_family_port_locator import Locator


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--original',type=Path,required=True);p.add_argument('--collection-root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--locator',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--family',choices=('sc','sfp'),default='sc')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(2);torch.set_num_threads(4)
    plan=json.loads(a.plan.read_text());old=[r for r in map(json.loads,(a.original/'frame_context.jsonl').open()) if r['family']==a.family]
    old_maps=np.load(a.original/'coarse_spatial_8x9.npy',mmap_mode='r');capacity=len(old)+len(plan['cases'])*200
    cache=np.lib.format.open_memmap(a.output/'coarse_spatial_8x9.npy',mode='w+',dtype=np.float16,shape=(capacity,3,256,8,9))
    context=(a.output/'frame_context.jsonl').open('w');index=0;groups={'train':set(),'validation':set()}
    for r in old:
        cache[index]=old_maps[r['feature_row']];r=dict(r,original_feature_row=r['feature_row'],feature_row=index,training_source='LLB_expert_layout' if a.family=='sc' else 'SFP_original_native')
        context.write(json.dumps(r,separators=(',',':'))+'\n');groups[r['split']].add(r['scene_group']);index+=1
    cache.flush();context.flush()
    model=Locator(None).cuda().eval();model.load_state_dict(torch.load(a.locator,map_location='cpu',weights_only=False)['model'])
    meta=torch.tensor([[1,0,1,0,0],[1,0,0,1,0],[1,0,0,0,1]] if a.family=='sc' else [[0,1,1,0,0],[0,1,0,1,0],[0,1,0,0,1]],device='cuda',dtype=torch.float32)
    seen=[];start=time.monotonic()
    for case in plan['cases']:
        labels=a.collection_root/case['case']/'native_labels';ready=labels/'summary.json'
        while not ready.exists():
            if time.monotonic()-start>6*3600:raise TimeoutError('Collection not complete; do not train on an unreported partial cache')
            time.sleep(20)
        rows=[json.loads(l) for l in (labels/'rows.jsonl').open()]
        for r in rows:
            if index>=capacity:raise ValueError('Unexpected capture density; preserve partial cache')
            rgb=[]
            for c in ('center','left','right'):
                image=cv2.imread(r['images'][c]);assert image.shape[:2]==(1024,1152)
                rgb.append(cv2.cvtColor(cv2.resize(image,(288,256),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB))
            image=torch.from_numpy(np.stack(rgb)).permute(0,3,1,2).cuda().float()/255
            with torch.inference_mode():point,flag,_,spatial=model(image,meta,return_features=True,return_spatial=True)
            cache[index]=F.adaptive_avg_pool2d(spatial,(8,9)).half().cpu().numpy()
            r=dict(r,feature_row=index,training_source='SC_rail_expansion' if a.family=='sc' else 'SFP_acquisition_gap',coarse_pixels_all_slots=(point.float().cpu().numpy()*[1151,1023]).tolist(),coarse_inframe_probabilities=flag.sigmoid().float().cpu().tolist())
            context.write(json.dumps(r,separators=(',',':'))+'\n');groups[r['split']].add(r['scene_group']);index+=1
        if groups['train']&groups['validation']:raise ValueError('Physical reset leakage')
        cache.flush();context.flush();seen.append(case['case'])
        (a.output/'progress.json').write_text(json.dumps(dict(completed=len(seen),planned=len(plan['cases']),valid_feature_rows=index,original_SC_rows=len(old),final_scenes_opened=False),indent=2)+'\n')
        print(json.dumps(dict(case=case['case'],new_rows=len(rows),completed=len(seen))),flush=True)
    context.close();cache.flush()
    summary=dict(family=a.family,valid_feature_rows=index,allocated_feature_rows=capacity,original_family_rows=len(old),new_family_rows=index-len(old),cases=seen,
        train_reset_count=len(groups['train']),validation_reset_count=len(groups['validation']),
        only_frame_context_rows_may_be_read=True,original_cache_modified=False,geometry_input=False,final_scenes_opened=False)
    if a.family=='sc':summary.update(original_SC_rows=len(old),new_SC_rows=index-len(old))
    (a.output/'cache_summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    from project_gpu_guard import selected_gpu_reservation
    with selected_gpu_reservation():main()
