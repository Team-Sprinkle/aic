#!/usr/bin/env python3
"""Fit SFP arrival heads on frozen RGB; select on development, audit once."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from train_sfp_arrival_probe import Probe


def events(rows,ids,prob,threshold=.95,streak=3):
    groups=defaultdict(list)
    for i in ids:
        r=rows[i];groups[r['episode']].append((r['time_s'],r['label'],float(prob[i]),
                              r['supervision_lateral_m'],r['supervision_axial_m'],
                              r['card'],r['port']))
    result=[]
    for episode,rr in groups.items():
        rr.sort();truth=[r[0] for r in rr if r[1]];run=0;hit=None
        for r in rr:
            run=run+1 if r[2]>=threshold else 0
            if run>=streak:hit=r;break
        outcome=('no_label' if not truth else 'missed' if hit is None else
                 'early' if hit[0]<truth[0] else 'correct' if hit[0]<=truth[-1] else 'late')
        result.append(dict(episode=episode,card=rr[0][5],port=rr[0][6],
            outcome=outcome,first_label_s=min(truth) if truth else None,
            trigger_s=hit[0] if hit else None,
            trigger_lateral_mm=hit[3]*1000 if hit else None,
            trigger_axial_mm=hit[4]*1000 if hit else None))
    return result


def counts(events):
    return dict(episodes=len(events),by_outcome=dict(Counter(x['outcome'] for x in events)),
        by_target={str(k):dict(Counter(x['outcome'] for x in events if (x['card'],x['port'])==k))
                   for k in sorted({(x['card'],x['port']) for x in events})})


def all_prob(model,image,state):
    model.eval()
    with torch.inference_mode():
        return np.concatenate([model(image[i:i+1024],state[i:i+1024]).sigmoid().cpu().numpy()
                               for i in range(0,len(image),1024)])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=800)
    p.add_argument('--device',default='cuda:1')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing output')
    a.output.mkdir(parents=True)
    random.seed(20261003);np.random.seed(20261003);torch.manual_seed(20261003)
    torch.set_num_threads(4)
    rows=[json.loads(s) for s in (a.cache/'rows.jsonl').open()]
    img=np.asarray(np.load(a.cache/'features.npy',mmap_mode='r'),np.float32)
    assert len(rows)==len(img)
    state=np.asarray([r['state']+r['task'] for r in rows],np.float32)
    split=np.asarray([r['split'] for r in rows]);label=np.asarray([r['label'] for r in rows])
    m=state[split=='train'].mean(0);s=state[split=='train'].std(0).clip(.01)
    state=(state-m)/s
    device=torch.device(a.device);image=torch.from_numpy(img).to(device)
    measured=torch.from_numpy(state).to(device);truth=torch.from_numpy(label.astype(np.float32)).to(device)
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        if r['split']=='train':groups[(r['card'],r['port'],r['label'])].append(i)
    assert len(groups)==20 and all(groups.values()),'All ten card/port settings need both classes'
    indices={key:np.asarray(value) for key,value in groups.items()}
    partitions={name:np.where(split==name)[0] for name in ('train','development','audit')}
    report=dict(schema='aic_sfp_balanced_arrival_probe/v1',cache=str(a.cache),
                rows=len(rows),episode_counts={name:len({rows[i]['episode'] for i in idx})
                                         for name,idx in partitions.items()},
                checkpoint_selection='development episodes only; audit read once after all checkpoints frozen',
                threshold=.95,streak=3,updates=a.updates,modes={},
                geometry_at_inference=False,actions_used=False,final_scenes_opened=False)
    for mode in ('state_only','image_only','image_state'):
        model=Probe(image.shape[1],measured.shape[1],mode).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
        best=None;history=[]
        for update in range(1,a.updates+1):
            q=[]
            for _ in range(64):
                card=random.randrange(5);port=random.randrange(2);y=random.randrange(2)
                q.append(int(np.random.choice(indices[(card,port,y)])))
            q=torch.tensor(q,device=device)
            model.train();opt.zero_grad(set_to_none=True)
            loss=nn.functional.binary_cross_entropy_with_logits(model(image[q],measured[q]),truth[q])
            loss.backward();opt.step()
            if update%100==0 or update==a.updates:
                prob=all_prob(model,image,measured)
                ev=events(rows,partitions['development'],prob)
                c=counts(ev)['by_outcome']
                score=100*c.get('early',0)+3*c.get('missed',0)+c.get('late',0)+c.get('no_label',0)
                point=dict(update=update,development=counts(ev),score=score,loss=float(loss.detach()))
                history.append(point);print(json.dumps(dict(mode=mode,**point)),flush=True)
                if best is None or score<best['score']:
                    best=point
                    torch.save(dict(model=model.state_dict(),mode=mode,state_mean=m,state_std=s,
                                    cache_manifest=str(a.cache/'manifest.json'),
                                    selection_update=update,threshold=.95,streak=3,
                                    final_scenes_opened=False),a.output/f'{mode}.pt')
        # The audit episodes are evaluated only after model selection is frozen.
        saved=torch.load(a.output/f'{mode}.pt',map_location=device,weights_only=False)
        model.load_state_dict(saved['model']);prob=all_prob(model,image,measured)
        audit=events(rows,partitions['audit'],prob)
        (a.output/f'{mode}_audit_episodes.json').write_text(json.dumps(audit,indent=2)+'\n')
        report['modes'][mode]=dict(best_development=best,audit=counts(audit),history=history)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
