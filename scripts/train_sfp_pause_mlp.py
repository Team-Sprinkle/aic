#!/usr/bin/env python3
"""Fit a small nonlinear pause head on frozen actor features and public state."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from handoff_region_prior import allows, load_prior
from train_temporal_transport_probe import LONG_AGES, TemporalTransport, build_data, predict, sha


def evaluate(head,features,rows,indices,prior,threshold=.95,streak=3):
    head.eval()
    with torch.inference_mode():
        probabilities=head(features[indices]).squeeze(-1).sigmoid().cpu().numpy()
    groups=defaultdict(list)
    for i,prob in zip(indices,probabilities):
        row=rows[int(i)]
        groups[row['episode']].append((row['time_s'],bool(row['handoff']),
            float(prob),bool(allows(prior,row['task'],row['state'][:3]))))
    summary=Counter();details=[]
    for episode,items in groups.items():
        items.sort()
        positives=[t for t,truth,_,_ in items if truth]
        if not positives:raise ValueError(f'No positive pause window: {episode}')
        begin,end=min(positives),max(positives)
        run=0;trigger=None
        for t,truth,prob,permitted in items:
            summary['raw_false_positive_rows']+=int(prob>=threshold and not truth)
            summary['permitted_false_positive_rows']+=int(prob>=threshold and permitted and not truth)
            run=run+1 if prob>=threshold and permitted else 0
            if run>=streak and trigger is None:trigger=t
        outcome=('missed' if trigger is None else 'early' if trigger<begin
                 else 'timely' if trigger<=end else 'late')
        summary[outcome]+=1
        details.append(dict(episode=episode,outcome=outcome,trigger_s=trigger,
                            window_start_s=begin,window_end_s=end))
    return dict(episodes=len(groups),**summary),details


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--actor',type=Path,required=True)
    p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=2000)
    p.add_argument('--hard-negative-mass',type=float,default=.25)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous output')
    if a.updates>2000 or not 0<a.hard_negative_mass<1:p.error('Bounded fit or mass exceeded')
    a.output.mkdir(parents=True)
    np.random.seed(20261007);torch.manual_seed(20261007);torch.set_num_threads(8)
    data=build_data(a.cache,a.base,'cuda',LONG_AGES)
    rows=data['rows'];saved=torch.load(a.actor,map_location='cpu',weights_only=False)
    old=saved['source_manifest']
    if not old['full_state_skip'] or not old['long_history'] or old['route_conditioning']:
        p.error('Expected frozen full-state action-only actor')
    model=TemporalTransport(data['mean'],data['std'],raw_state_skip=True,
                            feature_dim=data['features'].shape[1],state_task_dim=42,
                            route_conditioning=False).cuda().eval()
    model.load_state_dict(saved['model'])
    for q in model.parameters():q.requires_grad_(False)
    hidden=[]
    with torch.inference_mode():
        for chunk in np.array_split(np.arange(len(rows)),max(1,(len(rows)+255)//256)):
            ind=torch.as_tensor(chunk,device='cuda')
            hidden.append(predict(model,data,ind,False,return_hidden=True)[4].detach())
    latent=torch.cat(hidden)
    features=torch.cat((latent,data['state_task_full'].float()),dim=1)
    head=nn.Sequential(nn.LayerNorm(features.shape[1]),nn.Linear(features.shape[1],128),
                       nn.GELU(),nn.Linear(128,1)).cuda()
    opt=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-3)
    train=np.asarray([i for i,r in enumerate(rows) if r['family']=='sfp' and
                      r['split']=='train'],dtype=np.int64)
    positive=train[[bool(rows[i]['handoff']) for i in train]]
    negative=train[[not bool(rows[i]['handoff']) for i in train]]
    hard=negative[[bool(rows[i].get('past_transport_cutoff')) for i in negative]]
    ordinary=negative[[not bool(rows[i].get('past_transport_cutoff')) for i in negative]]
    if min(len(positive),len(hard),len(ordinary))<30:raise ValueError('Missing grouped head labels')
    val_original=np.asarray([i for i,r in enumerate(rows) if r['family']=='sfp' and
        r['split']=='validation' and int(r['episode'])<3100000],dtype=np.int64)
    val_tail=np.asarray([i for i,r in enumerate(rows) if r['family']=='sfp' and
        r['split']=='validation' and int(r['episode'])>=3100000],dtype=np.int64)
    prior=load_prior(a.prior)
    history=[];best=None;best_score=(float('inf'),float('inf'),float('inf'))
    n_hard=int(round(32*a.hard_negative_mass))
    for update in range(1,a.updates+1):
        idx=np.r_[np.random.choice(positive,32),
                  np.random.choice(hard,n_hard),
                  np.random.choice(ordinary,32-n_hard)]
        np.random.shuffle(idx)
        ind=torch.as_tensor(idx,device='cuda')
        head.train();opt.zero_grad(set_to_none=True)
        logits=head(features[ind]).squeeze(-1)
        loss=nn.functional.binary_cross_entropy_with_logits(logits,data['handoff'][ind])
        loss.backward();nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
        if update%100==0 or update==a.updates:
            original,original_details=evaluate(head,features,rows,val_original,prior)
            tail,tail_details=evaluate(head,features,rows,val_tail,prior)
            score=(original.get('early',0)+original.get('missed',0)+original.get('late',0),
                   tail.get('permitted_false_positive_rows',0),
                   tail.get('raw_false_positive_rows',0))
            item=dict(update=update,loss=float(loss.detach()),original=original,tail=tail)
            history.append(item)
            print(json.dumps(item),flush=True)
            if score<best_score:
                best_score=score;best=dict(item,original_details=original_details,
                                          tail_details=tail_details)
                torch.save(dict(schema='aic_sfp_pause_mlp/v1',model=head.state_dict(),
                                input_dim=features.shape[1],update=update,
                                actor=str(a.actor),actor_sha256=sha(a.actor),
                                cache=str(a.cache),prior=str(a.prior),
                                threshold=.95,streak=3,final_scenes_opened=False),
                           a.output/'best.pt')
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    summary=dict(schema='aic_sfp_pause_mlp_training/v1',best=best,
                 action_weights_frozen=True,train_episodes=len({rows[i]['episode'] for i in train}),
                 original_validation_episodes=len({rows[i]['episode'] for i in val_original}),
                 tail_validation_episodes=len({rows[i]['episode'] for i in val_tail}),
                 train_positive_rows=len(positive),train_ordinary_negative_rows=len(ordinary),
                 train_post_cutoff_negative_rows=len(hard),hard_negative_mass=a.hard_negative_mass,
                 automatic_handoff_enabled=False,final_scenes_opened=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('best',)}
                     |dict(best=best['update'],original=best['original'],tail=best['tail'])),flush=True)


if __name__=='__main__':main()
