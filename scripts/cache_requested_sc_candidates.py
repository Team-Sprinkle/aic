#!/usr/bin/env python3
"""Freeze observation-only counterfactual requested-port crop proposals.

Both public queries are run offline to create wrong-request training examples.
Deployment still requests one port. No target geometry is read by inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from train_sc_requested_history_locator import RequestedHistoryLocator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    rows=[json.loads(l) for l in (a.cache/'frame_context.jsonl').open()]
    complete=json.loads((a.cache/'cache_summary.json').read_text())
    assert len(rows)==complete['valid_feature_rows']
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if saved['use_history']:raise ValueError('This bank builder requires the selected current-only head')
    model=RequestedHistoryLocator(False).cuda().eval();model.load_state_dict(saved['model'])
    maps=np.load(a.cache/'coarse_spatial_8x9.npy',mmap_mode='r')
    points=np.empty((len(rows),2,3,2),np.float32);flags=np.empty((len(rows),2,3),np.float32)
    with torch.inference_mode():
        for start in range(0,len(rows),64):
            batch=rows[start:start+64];x=torch.from_numpy(np.stack([maps[r['feature_row']] for r in batch])).cuda()
            history=x.new_zeros((len(batch),10,3,6,256));motion=x.new_zeros((len(batch),10,9))
            mask=torch.ones(len(batch),10,dtype=torch.bool,device='cuda')
            for port in [0,1]:
                point,flag,_=model(x,history,motion,mask,torch.full((len(batch),),port,device='cuda'))
                points[start:start+len(batch),port]=(point*point.new_tensor([1151,1023])).cpu().numpy()
                flags[start:start+len(batch),port]=flag.sigmoid().cpu().numpy()
    np.save(a.output/'requested_candidates.npy',points);np.save(a.output/'projected_inframe_probabilities.npy',flags)
    manifest=[dict(episode=r['episode'],time_s=r['time_s'],feature_row=r['feature_row']) for r in rows]
    (a.output/'row_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
    (a.output/'summary.json').write_text(json.dumps(dict(rows=len(rows),geometry_input=False,
        checkpoint=str(a.checkpoint),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        frame_context_sha256=hashlib.sha256((a.cache/'frame_context.jsonl').read_bytes()).hexdigest(),
        deployment_predicts_one_requested_port=True,offline_counterfactual_queries=[0,1],final_scenes_opened=False),indent=2)+'\n')


if __name__=='__main__':main()
