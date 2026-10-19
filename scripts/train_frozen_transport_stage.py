#!/usr/bin/env python3
"""Learn SC route phase without changing a frozen action policy."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from train_temporal_transport_probe import LONG_AGES, TemporalTransport, build_data, predict


def evaluate(model, data, indices):
    grouped=defaultdict(list);all_true=[];all_pred=[]
    model.eval()
    with torch.inference_mode():
        for part in np.array_split(indices,max(1,(len(indices)+255)//256)):
            ind=torch.as_tensor(part,device='cuda')
            stage=predict(model,data,ind,False)[1].softmax(-1).cpu().numpy()
            for i,p in zip(part,stage):
                r=data['rows'][int(i)]
                grouped[r['episode']].append((r['time_s'],r['stage'],float(p[5])))
                all_true.append(r['stage']);all_pred.append(int(p.argmax()))
    early=correct=missed=0
    for seq in grouped.values():
        seq.sort();start=min(t for t,s,_ in seq if s==5)
        run=0;trigger=None
        for t,_,p in seq:
            run=run+1 if p>=.9 else 0
            if run>=3:trigger=t;break
        if trigger is None:missed+=1
        elif trigger<start:early+=1
        else:correct+=1
    truth=np.asarray(all_true);pred=np.asarray(all_pred)
    return {'accuracy':float(np.mean(truth==pred)),
        'stage5_precision':float(np.mean(truth[pred==5]==5)) if np.any(pred==5) else 0.,
        'stage5_recall':float(np.mean(pred[truth==5]==5)),
        'stage5_false_positive_rows':int(np.sum((pred==5)&(truth!=5))),
        'early_stage5_episodes':early,'correct_stage5_episodes':correct,
        'missed_stage5_episodes':missed}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--actor',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=1500)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing output')
    a.output.mkdir(parents=True)
    random.seed(20261001);np.random.seed(20261001);torch.manual_seed(20261001);torch.set_num_threads(8)
    saved=torch.load(a.actor,map_location='cpu',weights_only=False)
    manifest=dict(saved['source_manifest'])
    if not manifest.get('zero_previous_command') or not manifest.get('long_history'):
        p.error('Expected zero-previous long-history actor')
    data=build_data(a.cache,a.base,'cuda',LONG_AGES);data['prev'].zero_()
    model=TemporalTransport(data['mean'],data['std'],raw_state_skip=True,
        feature_dim=data['features'].shape[1],state_task_dim=42,route_conditioning=False).cuda()
    model.load_state_dict(saved['model'])
    for name,param in model.named_parameters():param.requires_grad_(name.startswith('stage.'))
    opt=torch.optim.AdamW(model.stage.parameters(),lr=1e-3,weight_decay=1e-3)
    rows=data['rows'];stages={s:np.asarray([i for i,r in enumerate(rows) if r['family']=='sc' and
        r['split']=='train' and r['stage']==s]) for s in range(6)}
    val=np.asarray([i for i,r in enumerate(rows) if r['family']=='sc' and r['split']=='validation'])
    manifest.update({'stage_aux_loss':1.,'stage_training':
        'frozen action/temporal backbone; only linear stage head trained',
        'stage_train_rows':{str(k):len(v) for k,v in stages.items()},
        'stage_validation_rows':len(val),'stage_update_cap':a.updates,
        'stage_actor_checkpoint':str(a.actor)})
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    best=float('inf');best_update=0;history=[]
    for update in range(1,a.updates+1):
        selection=np.r_[*[np.random.choice(stages[s],12) for s in range(6)]]
        ind=torch.as_tensor(selection,device='cuda')
        model.eval();opt.zero_grad(set_to_none=True)
        stage=predict(model,data,ind,False)[1]
        loss=nn.functional.cross_entropy(stage,data['stage'][ind]);loss.backward();opt.step()
        if update%100==0 or update==a.updates:
            metrics=evaluate(model,data,val)
            score=(metrics['early_stage5_episodes']*100+
                   metrics['missed_stage5_episodes']+(1-metrics['accuracy'])*5)
            history.append({'update':update,'train_loss':float(loss.detach()),
                'event_metrics':metrics,'score':score})
            print(json.dumps(history[-1]),flush=True)
            if score<best:
                best=score;best_update=update
                torch.save({'model':model.state_dict(),'update':update,
                    'history_enabled':True,'source_manifest':manifest,
                    'final_scenes_opened':False,'rl_started':False},a.output/'best.pt')
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    (a.output/'summary.json').write_text(json.dumps({'best_update':best_update,
        'best_event_metrics':next(r['event_metrics'] for r in history if r['update']==best_update),
        'action_weights_frozen':True,'final_scenes_opened':False},indent=2)+'\n')


if __name__=='__main__':main()
