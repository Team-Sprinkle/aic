#!/usr/bin/env python3
"""Replay actual public RGB history; query the SAME candidate for both SC ports.

No geometry is used for either prediction. This diagnoses whether a purported
identity check rejects a conflicting public request, not insertion performance.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch

from native_candidate_association import NativeCandidateAssociation
from train_two_family_port_locator import Locator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--locator',type=Path,required=True)
    p.add_argument('--association',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(2);torch.set_num_threads(4)
    d=json.loads((a.audit/'scored_stream.json').read_text())
    frames={c:{stamp:path for stamp,path in rows} for c,rows in d['cameras'].items()}
    model=Locator(None).cuda().eval();model.load_state_dict(torch.load(a.locator,map_location='cpu',weights_only=False)['model'])
    observer=NativeCandidateAssociation(a.association)
    meta=torch.tensor([[1,0,1,0,0],[1,0,0,1,0],[1,0,0,0,1]],device='cuda',dtype=torch.float32)
    results=[]
    events=[json.loads(line) for line in (a.audit/'runtime_trace.jsonl').open()]
    # Inspection is logged inside predict_chunk, before its enclosing proposal
    # record. Join by decision identity, not log line order or previous state.
    inspections={}
    for e in events:
        if e['event']=='rule_visual_inspection':
            inspections.setdefault(e['sim_time'],[]).append(e)
    with torch.inference_mode():
        for e in events:
            if e['event']=='proposal':
                state=e['state'];images=[]
                for c in ('center','left','right'):
                    image=cv2.imread(frames[c][e['image_sim_ns'][c]])
                    images.append(cv2.cvtColor(cv2.resize(image,(288,256),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB))
                image=torch.from_numpy(np.stack(images)).permute(0,3,1,2).cuda().float()/255
                _,_,_,spatial=model(image,meta,return_features=True,return_spatial=True)
                observer.observe(spatial,state)
            else:continue
            for inspection in inspections.get(e['sim_time'],[]):
                r=inspection['report']
                if 'candidate_association_probability' not in r:continue
                observer.observe(spatial,state)
                points=r['coarse_pixels'];requested=inspection['requested_port']
                scores={str(port):observer.probabilities(points,port).tolist() for port in (0,1)}
                row=dict(sim_time=e['sim_time'],requested_port=requested,reported=r['candidate_association_probability'],
                         replay_scores=scores,reason=r['reason'],valid=r['valid'],state=state[:7],
                         normalized_state=((torch.tensor(observer.state,device=observer.device)-observer.mean)/observer.std).cpu().tolist(),
                         requested_minus_other_min=float(min(np.asarray(scores[str(requested)])-scores[str(1-requested)])),
                         geometry_input=False)
                results.append(row)
    (a.output/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
    summary=dict(inspections=len(results),accepted=[r for r in results if r['valid']],
                 non_discriminating_both_requests=sum(min(r['replay_scores']['0'])>=.9 and min(r['replay_scores']['1'])>=.9 for r in results),
                 exact_current_frame_and_public_history=True,geometry_input=False,final_scenes_opened=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
