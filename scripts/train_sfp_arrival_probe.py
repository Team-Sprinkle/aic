#!/usr/bin/env python3
"""Bounded SFP arrival diagnostic on frozen cached RGB features.

Oracle port-frame TCP geometry labels training/evaluation rows only. At inference
the model receives frozen RGB features, measured robot state, and public task code.
This is a stop proposal probe, not a port-identity or insertion verifier.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn


def read_data(cache, source):
    rows=[json.loads(s) for s in (cache/'rows.jsonl').open()]
    oracle={}
    for s in source.open():
        x=json.loads(s)
        oracle[(x['episode_index'],x['frame_index'])]=x['observed_tcp_pose_port_frame']
    ids=[]; labels=[]; xyz=[]
    for i,r in enumerate(rows):
        if r['family']!='sfp': continue
        p=oracle[(r['episode'],r['frame'])][:3]
        lateral=float(np.hypot(p[0],p[1])); z=float(p[2])
        # Entry into a buffered local-actor region, before contact/insertion.
        target=int(-.10<=z<=-.07 and lateral<=.01)
        ids.append(i); labels.append(target); xyz.append((lateral,z))
    ids=np.asarray(ids);labels=np.asarray(labels);xyz=np.asarray(xyz)
    ximage=np.asarray(np.load(cache/'features.npy',mmap_mode='r')[ids],dtype=np.float32)
    xstate=np.asarray([rows[i]['state']+rows[i]['task'] for i in ids],dtype=np.float32)
    train=np.asarray([rows[i]['split']=='train' for i in ids]);val=~train
    mean=xstate[train].mean(0);std=xstate[train].std(0).clip(.01)
    xstate=(xstate-mean)/std
    return rows,ids,ximage,xstate,labels,xyz,train,val,mean,std


class Probe(nn.Module):
    def __init__(self,image_dim,state_dim,mode):
        super().__init__();self.mode=mode
        input_dim=(image_dim if mode!='state_only' else 0)+(state_dim if mode!='image_only' else 0)
        self.head=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,128),nn.GELU(),
                                nn.Linear(128,1))

    def forward(self,image,state):
        x=(state if self.mode=='state_only' else image if self.mode=='image_only' else
           torch.cat((image,state),-1))
        return self.head(x).squeeze(-1)


def episode_metrics(rows,ids,labels,xyz,probs,threshold=.95,streak=3):
    groups=defaultdict(list)
    for k,i in enumerate(ids):
        r=rows[i];groups[(r['split'],r['episode'])].append((r['time_s'],int(labels[k]),
                             float(probs[k]),float(xyz[k,0]),float(xyz[k,1])))
    by_split={};episodes=[]
    for (split,episode),rr in groups.items():
        rr.sort();truth=[t for t,y,*_ in rr if y]
        if not truth: outcome='no_label';trigger=None
        else:
            n=0;trigger=None
            for t,y,p,*_ in rr:
                n=n+1 if p>=threshold else 0
                if n>=streak:trigger=t;break
            outcome=('missed' if trigger is None else 'early' if trigger<truth[0] else
                     'correct' if trigger<=truth[-1] else 'late')
        episodes.append(dict(split=split,episode=episode,outcome=outcome,trigger_s=trigger,
                             true_first_s=min(truth) if truth else None,
                             true_last_s=max(truth) if truth else None,
                             rows=len(rr)))
    for split in ('train','validation'):
        es=[x for x in episodes if x['split']==split]
        by_split[split]=dict(episodes=len(es),counts={k:sum(x['outcome']==k for x in es)
                             for k in ('correct','early','late','missed','no_label')})
    return by_split,episodes


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=400)
    p.add_argument('--device',default='cuda:1')
    a=p.parse_args()
    if a.output.exists():p.error('Output already exists')
    a.output.mkdir(parents=True)
    random.seed(20261003);np.random.seed(20261003);torch.manual_seed(20261003)
    torch.set_num_threads(4)
    rows,ids,image,state,labels,xyz,train,val,mean,std=read_data(a.cache,a.source)
    device=torch.device(a.device)
    image=torch.from_numpy(image).to(device);state=torch.from_numpy(state).to(device)
    truth=torch.from_numpy(labels.astype(np.float32)).to(device)
    result={'schema':'aic_sfp_arrival_probe/v1','source':str(a.source),'cache':str(a.cache),
            'label':'port-frame measured TCP axial z in [-100,-70] mm and lateral <=10 mm',
            'oracle_at_inference':False,'threshold':.95,'streak':3,
            'rows':len(ids),'train_rows':int(train.sum()),'validation_rows':int(val.sum()),
            'train_positive':int(labels[train].sum()),'validation_positive':int(labels[val].sum()),
            'final_scenes_opened':False,'modes':{}}
    for mode in ('state_only','image_only','image_state'):
        model=Probe(image.shape[1],state.shape[1],mode).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
        pos=np.where(train&(labels==1))[0];neg=np.where(train&(labels==0))[0]
        best=None;hist=[]
        for update in range(1,a.updates+1):
            choices=np.r_[np.random.choice(pos,64),np.random.choice(neg,64)]
            q=torch.from_numpy(choices).to(device)
            model.train();opt.zero_grad(set_to_none=True)
            loss=nn.functional.binary_cross_entropy_with_logits(model(image[q],state[q]),truth[q])
            loss.backward();opt.step()
            if update%100==0 or update==a.updates:
                model.eval()
                with torch.no_grad():
                    probs=np.concatenate([model(image[j:j+1024],state[j:j+1024]).sigmoid().cpu().numpy()
                                          for j in range(0,len(ids),1024)])
                counts,events=episode_metrics(rows,ids,labels,xyz,probs)
                score=(counts['validation']['counts']['early']*100+
                       counts['validation']['counts']['missed']*3+
                       counts['validation']['counts']['late'])
                record=dict(update=update,loss=float(loss.detach()),score=score,episodes=counts)
                hist.append(record);print(json.dumps(dict(mode=mode,**record)),flush=True)
                if best is None or score<best[0]:
                    best=(score,record,events)
                    torch.save(dict(model=model.state_dict(),mode=mode,state_mean=mean,
                                    state_std=std,threshold=.95,streak=3,
                                    label=result['label'],feature_dim=image.shape[1],
                                    final_scenes_opened=False),a.output/f'{mode}.pt')
        result['modes'][mode]=dict(best=best[1],history=hist)
        (a.output/f'{mode}_episodes.json').write_text(json.dumps(best[2],indent=2)+'\n')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
