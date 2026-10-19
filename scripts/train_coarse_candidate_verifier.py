#!/usr/bin/env python3
"""Train proposal-conditioned requested-port identity/crop verifier.

The label checks a candidate against independent simulator port coordinates.
It does NOT certify visual visibility or safe stop; force is absent here.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_distance_coarse_crop import CoarseCrop, batch, distance_histories
from train_requested_history_crop import CAMERAS, HEIGHT, LOOKUP, WIDTH, rows_from


class CandidateVerifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.view = nn.Sequential(nn.Linear(128+128+2+7,128),nn.GELU(),
                                  nn.Linear(128,64),nn.GELU())
        self.final = nn.Sequential(nn.Linear(128+7,64),nn.GELU(),nn.Linear(64,1))

    def forward(self, context, maps, candidate, state):
        b=maps.shape[0]
        # A 5x5 patch roughly covers the 160 px original-image crop.
        offsets=torch.linspace(-1.8,1.8,5,device=maps.device)
        yy,xx=torch.meshgrid(offsets,offsets,indexing='ij')
        delta=torch.stack((xx/35,yy/31),-1)[None,None]
        grid=(candidate*2-1)[:,:,None,None,:]+2*delta
        grid=grid.reshape(b*3,5,5,2)
        sampled=F.grid_sample(maps.reshape(b*3,128,32,36),grid,
                              align_corners=True,padding_mode='zeros')
        local=sampled.mean((-1,-2)).reshape(b,3,128)
        x=torch.cat((context,local,candidate,state[:,None].expand(-1,3,-1)), -1)
        views=self.view(x)
        pooled=torch.cat((views.mean(1),views.amin(1),state),-1)
        return self.final(pooled).squeeze(-1)


def actual(rows, indices, names):
    targets=[];others=[]
    for i,name in zip(indices,names):
        row=rows[i]
        t=next((t for t in row['targets'] if t['identity']==name),None)
        targets.append(np.asarray([t['cameras'][cam][:2] for cam in CAMERAS],np.float32)
                       if t and all(t['cameras'][cam][3] for cam in CAMERAS)
                       else None)
        alternatives=[]
        for other in row['targets']:
            if other['identity']==name:continue
            if all(other['cameras'][cam][3] for cam in CAMERAS):
                alternatives.append(np.asarray([other['cameras'][cam][:2]
                                                for cam in CAMERAS],np.float32))
        others.append(alternatives)
    return targets,others


def labels(candidate, targets, alternatives):
    # Candidate and references are original-image pixel coordinates.
    out=[]
    for p,t,other in zip(candidate,targets,alternatives):
        if t is None:out.append(0.);continue
        error=np.linalg.norm(p-t,axis=-1)
        nearest=np.min(np.asarray([np.linalg.norm(p-q,axis=-1) for q in other]),axis=0) if other else np.full(3,np.inf)
        out.append(float(np.max(error)<=60 and np.all(error<nearest)))
    return np.asarray(out,np.float32)


def train_candidates(pred, targets, others):
    b=len(pred);p=pred.copy()*[WIDTH-1,HEIGHT-1]
    for k in range(b):
        roll=random.random()
        if roll<.3 and targets[k] is not None:
            p[k]=targets[k]+np.random.randn(3,2).astype(np.float32)*15
        elif roll<.6 and others[k]:
            p[k]=random.choice(others[k])+np.random.randn(3,2).astype(np.float32)*10
        elif roll<.75:
            p[k]=np.random.rand(3,2).astype(np.float32)*[WIDTH-1,HEIGHT-1]
        elif roll<.9 and targets[k] is not None:
            p[k]=targets[k]+np.random.randn(3,2).astype(np.float32)*45
        # Remaining predictions follow the actual crop head distribution.
    return np.clip(p,[0,0],[WIDTH-1,HEIGHT-1]).astype(np.float32)


@torch.no_grad()
def evaluate(crop, head, rows, full, small, histories, indices, device, output=None):
    crop.eval();head.eval();records=[]
    for off in range(0,len(indices),8):
        chosen=indices[off:off+8]
        data=batch(rows,full,small,histories,chosen,device)
        xy,availability,_,context,maps=crop(*data[:5],state=data[7],return_context=True)
        probs=head(context,maps,xy.float(),data[7]).sigmoid().cpu().numpy()
        pred=xy.float().cpu().numpy()*[WIDTH-1,HEIGHT-1]
        names=[rows[i]['selected_identity'] for i in chosen]
        targets,others=actual(rows,chosen,names)
        y=labels(pred,targets,others)
        for k,i in enumerate(chosen):
            r=rows[i]
            records.append({'episode':r['episode'],'time_s':r['time_s'],
                            'requested':names[k],'score':float(probs[k]),
                            'correct_crop_label_only':bool(y[k]),
                            'prediction_px':pred[k].tolist(),
                            'inframe_score':availability[k].sigmoid().cpu().tolist()})
    if output:
        with open(output,'w') as f:
            for r in records:f.write(json.dumps(r)+'\n')
    return records


def thresholds(records):
    # Freeze the lowest threshold with zero wrong accepted on grouped dev.
    values=sorted({0.,1.}|{r['score'] for r in records})
    selected=1.
    for t in values:
        acc=[r for r in records if r['score']>=t]
        if all(r['correct_crop_label_only'] for r in acc):
            selected=t;break
    accepted=[r for r in records if r['score']>=selected]
    by_ep=defaultdict(list)
    for r in accepted:by_ep[r['episode']].append(r)
    return selected,{'threshold':selected,'accepted':len(accepted),
                     'wrong_accepted':sum(not r['correct_crop_label_only'] for r in accepted),
                     'episodes_with_accept':len(by_ep)}


def report(records,threshold):
    accepted=[r for r in records if r['score']>=threshold]
    return {'rows':len(records),'correct_proposals':sum(r['correct_crop_label_only'] for r in records),
            'accepted':len(accepted),'wrong_accepted':sum(not r['correct_crop_label_only'] for r in accepted),
            'episodes_with_accept':len({r['episode'] for r in accepted}),
            'threshold_from_development':threshold}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--labels',type=Path,required=True)
    ap.add_argument('--stress-labels',type=Path,required=True)
    ap.add_argument('--feature-cache',type=Path,required=True)
    ap.add_argument('--crop-checkpoint',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--updates',type=int,default=1000)
    ap.add_argument('--minutes',type=float,default=60)
    a=ap.parse_args()
    if a.output.exists():ap.error('Output exists')
    a.output.mkdir(parents=True)
    random.seed(20261005);np.random.seed(20261005);torch.manual_seed(20261005)
    torch.set_num_threads(8)
    rows=rows_from(a.labels)+rows_from(a.stress_labels)
    original=len(rows_from(a.labels))
    split={k:[i for i,r in enumerate(rows) if r['split']==k and (k!='validation' or i<original)]
           for k in ('train','validation')}
    stress=list(range(original,len(rows)))
    groups={name:{rows[i]['scene_group'] for i in indices}
            for name,indices in [('train',split['train']),('dev',split['validation']),('stress',stress)]}
    if groups['train']&groups['dev'] or groups['train']&groups['stress'] or groups['dev']&groups['stress']:
        raise ValueError('Scene overlap')
    full=np.load(a.feature_cache/'features.npy',mmap_mode='r')
    small=np.load(a.feature_cache/'distance_4x5_features.npy',mmap_mode='r')
    expected=[{'episode':r['episode'],'time_s':r['time_s'],'scene_group':r['scene_group'],'images':r['images']} for r in rows]
    if json.loads((a.feature_cache/'features_manifest.json').read_text())!=expected:
        raise ValueError('Feature cache/row mismatch')
    histories=distance_histories(rows)
    ckpt=torch.load(a.crop_checkpoint,map_location='cpu',weights_only=False)
    crop=CoarseCrop(not ckpt['history_off'],ckpt.get('state_conditioned',False)).cuda().eval()
    missing=crop.load_state_dict(ckpt['model'],strict=False)
    if missing.unexpected_keys or (missing.missing_keys and
        (ckpt.get('state_conditioned',False) or
         any(not key.startswith('state.') for key in missing.missing_keys))):
        raise ValueError(f'Unexpected crop checkpoint mismatch: {missing}')
    for p in crop.parameters():p.requires_grad_(False)
    head=CandidateVerifier().cuda();opt=torch.optim.AdamW(head.parameters(),lr=3e-4)
    (a.output/'manifest.json').write_text(json.dumps({'crop_checkpoint':str(a.crop_checkpoint),
        'train_groups':sorted(groups['train']),'dev_groups':sorted(groups['dev']),
        'inspected_stress_groups':sorted(groups['stress']),
        'actual_visibility_labels':False,'safe_stop_labels':False,
        'final_scenes_opened':False},indent=2)+'\n')
    best=-1;best_step=0;logs=[];start=time.monotonic()
    for step in range(1,a.updates+1):
        if time.monotonic()-start>a.minutes*60:break
        chosen=random.choices(split['train'],k=16)
        data=batch(rows,full,small,histories,chosen,'cuda',True)
        with torch.no_grad():xy,_,_,context,maps=crop(*data[:5],state=data[7],return_context=True)
        names=[]
        for i,t in zip(chosen,data[4].tolist()):
            names.append(next(k for k,v in LOOKUP.items() if v==t))
        targets,others=actual(rows,chosen,names)
        cand=train_candidates(xy.float().cpu().numpy(),targets,others)
        y=labels(cand,targets,others)
        cand=torch.from_numpy(cand/np.asarray([WIDTH-1,HEIGHT-1],np.float32)).cuda()
        target=torch.from_numpy(y).cuda()
        head.train();opt.zero_grad(set_to_none=True)
        logits=head(context.detach(),maps.detach(),cand,data[7])
        loss=F.binary_cross_entropy_with_logits(logits,target)
        loss.backward();nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
        if step%100==0:
            dev=evaluate(crop,head,rows,full,small,histories,split['validation'],'cuda')
            threshold,summary=thresholds(dev)
            score=summary['accepted']
            logs.append({'step':step,'loss':float(loss.detach()),'dev':summary})
            print(json.dumps(logs[-1]),flush=True)
            if score>best:
                best=score;best_step=step
                torch.save({'model':head.state_dict(),'step':step},a.output/'best.pt')
    if not best_step:raise RuntimeError('No checkpoint')
    head.load_state_dict(torch.load(a.output/'best.pt',weights_only=False)['model'])
    dev=evaluate(crop,head,rows,full,small,histories,split['validation'],'cuda',a.output/'dev_predictions.jsonl')
    threshold,dev_summary=thresholds(dev)
    stress_rows=evaluate(crop,head,rows,full,small,histories,stress,'cuda',a.output/'stress_predictions.jsonl')
    result={'schema':'coarse_candidate_verifier_diagnostic_v1','best_step':best_step,
            'updates':step,'dev':report(dev,threshold),
            'inspected_stress':report(stress_rows,threshold),
            'true_visibility_trained':False,'safe_stop_trained':False,'promotion':False}
    (a.output/'history.json').write_text(json.dumps(logs,indent=2)+'\n')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
