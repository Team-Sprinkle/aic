#!/usr/bin/env python3
"""Bounded sparse-card occupancy repair with frozen visual/endpoint backbone.

CAD rail projections are loaded for scoring only. The optimizer sees reliable
scene-file occupancy labels and original RGB; it cannot move projected rail
endpoints to fit an ambiguous mark.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from train_sfp_board_reference import BoardReferenceLocal, RailViews, evaluate


def groups_by_count(rows):
    grouped=defaultdict(lambda:defaultdict(list))
    for i,row in enumerate(rows):
        grouped[row['card_count']][row['scene_group']].append(i)
    return grouped


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--init',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=1000)
    p.add_argument('--minutes',type=float,default=30.)
    p.add_argument('--batch-size',type=int,default=4)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing output')
    if a.updates>1000 or a.minutes>30 or a.batch_size>8:p.error('Bounded fit exceeded')
    a.output.mkdir(parents=True)
    rows=[json.loads(line) for line in (a.labels/'rows.jsonl').open()]
    development=[i for i,r in enumerate(rows) if r['split']=='validation']
    raw_train=[i for i,r in enumerate(rows) if r['split']=='train']
    grouped=groups_by_count([rows[i] for i in raw_train])
    held=set()
    for count in range(1,6):
        scenes=sorted(grouped[count],key=lambda s:hashlib.sha256(s.encode()).hexdigest())
        if len(scenes)<8:raise ValueError(f'Only {len(scenes)} train scenes for card count {count}')
        held.update(scenes[:4])
    test=[i for i in raw_train if rows[i]['scene_group'] in held]
    train=[i for i in raw_train if rows[i]['scene_group'] not in held]
    train_scenes={rows[i]['scene_group'] for i in train}
    dev_scenes={rows[i]['scene_group'] for i in development}
    if train_scenes & held or dev_scenes & (train_scenes|held):
        raise ValueError('Scene leakage')
    torch.manual_seed(20261007);np.random.seed(20261007);rng=random.Random(20261007)
    torch.set_num_threads(4)
    saved=torch.load(a.init,map_location='cpu',weights_only=False)
    if saved.get('head')!='slot_local':raise ValueError('Expected slot-local board model')
    model=BoardReferenceLocal(saved['init'])
    model.load_state_dict(saved['model'])
    for name,param in model.named_parameters():
        param.requires_grad_(name.startswith('slot_'))
    model.cuda().eval()
    optimizer=torch.optim.AdamW((x for x in model.parameters() if x.requires_grad),
                                lr=2e-4,weight_decay=1e-4)
    def loader(indices):
        return DataLoader(RailViews(rows,indices),batch_size=a.batch_size,
                          shuffle=False,num_workers=2,pin_memory=True,
                          multiprocessing_context='spawn')
    dev_loader=loader(development)
    test_loader=loader(test)
    baseline_dev=evaluate(model,dev_loader,rows,'cuda',a.output/'baseline_development_predictions.jsonl')
    # Keep test sealed until one checkpoint has been selected on development.
    by_count=defaultdict(lambda:defaultdict(list))
    for i in train:
        r=rows[i];by_count[r['card_count']][r['scene_group']].append(i)
    scene_names={c:tuple(sorted(v)) for c,v in by_count.items()}
    history=[];best_score=(baseline_dev['confident_empty_slot_errors'],
                           -baseline_dev['exact_occupancy_episodes'])
    best_update=0;started=time.monotonic();reason='update_cap'
    for update in range(1,a.updates+1):
        if time.monotonic()-started>a.minutes*60:
            reason='wall_budget';break
        chosen=[]
        for _ in range(a.batch_size):
            count=rng.choice(tuple(sorted(by_count)))
            scene=rng.choice(scene_names[count])
            chosen.append(rng.choice(by_count[count][scene]))
        batch=RailViews(rows,chosen)
        samples=[batch[j] for j in range(len(chosen))]
        images=torch.stack([s[0] for s in samples]).cuda(non_blocking=True).float()/255.
        occupied=torch.stack([s[3] for s in samples]).cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            _,_,logits,_=model(images)
            loss=nn.functional.binary_cross_entropy_with_logits(logits.float(),occupied)
        loss.backward()
        nn.utils.clip_grad_norm_((x for x in model.parameters() if x.requires_grad),1.)
        optimizer.step()
        if update%100==0 or update==a.updates:
            dev=evaluate(model,dev_loader,rows,'cuda',a.output/f'development_{update:04d}.jsonl')
            score=(dev['confident_empty_slot_errors'],-dev['exact_occupancy_episodes'])
            item=dict(update=update,elapsed_min=(time.monotonic()-started)/60,
                      train_loss=float(loss.detach()),development=dev)
            history.append(item)
            print(json.dumps({k:item[k] for k in ('update','elapsed_min','train_loss')}
                             |dict(development_exact=dev['exact_occupancy_episodes'],
                                   development_false_empty=dev['confident_empty_slot_errors'])),flush=True)
            if score<best_score:
                best_score=score;best_update=update
                torch.save(dict(model=model.state_dict(),init=saved['init'],head='slot_local',
                                update=update,source_checkpoint=str(a.init),
                                final_scenes_opened=False),a.output/'best.pt')
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    if best_update:
        model.load_state_dict(torch.load(a.output/'best.pt',map_location='cpu',weights_only=False)['model'])
    test_result=evaluate(model,test_loader,rows,'cuda',a.output/'selected_test_predictions.jsonl')
    base=BoardReferenceLocal(saved['init']);base.load_state_dict(saved['model']);base.cuda().eval()
    baseline_test=evaluate(base,test_loader,rows,'cuda',a.output/'baseline_test_predictions.jsonl')
    summary=dict(schema='aic_sfp_diverse_occupancy_head/v1',
                 source_checkpoint=str(a.init),labels=str(a.labels),
                 train_groups=len(train_scenes),development_groups=len(dev_scenes),
                 untouched_test_groups=len(held),test_scenes=sorted(held),
                 action_labels_used=False,geometry_runtime_input=False,
                 endpoints_and_backbone_frozen=True,
                 selected_update=best_update,stop_reason=reason,
                 baseline_development=baseline_dev,
                 selected_development=(next((h['development'] for h in history
                                            if h['update']==best_update),baseline_dev)),
                 baseline_test=baseline_test,selected_test=test_result,
                 gate=dict(development_exact_min=16,test_exact_min=18,
                           confident_empty_slot_errors_max=0,
                           passed=bool(best_update and
                               (next(h['development'] for h in history if h['update']==best_update)
                                ['exact_occupancy_episodes']>=16) and
                               test_result['exact_occupancy_episodes']>=18 and
                               test_result['confident_empty_slot_errors']==0)),
                 final_scenes_opened=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(selected_update=best_update,
                          baseline_test_exact=baseline_test['exact_occupancy_episodes'],
                          selected_test_exact=test_result['exact_occupancy_episodes'],
                          passed=summary['gate']['passed'])),flush=True)


if __name__=='__main__':main()
