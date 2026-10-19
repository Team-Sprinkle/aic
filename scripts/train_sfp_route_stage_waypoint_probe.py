#!/usr/bin/env python3
"""Bounded observation-only SFP route-stage and next-waypoint probe."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from train_temporal_transport_probe import LONG_AGES, TemporalTransport, build_data, predict, sha


def metrics(rows, indices, stages, waypoints, stage_logits, predicted_xyz):
    actual_stage = stages[indices]
    actual_xyz = waypoints[indices, :3]
    guessed_stage = stage_logits.argmax(-1)
    mm = np.linalg.norm((predicted_xyz - actual_xyz) * 1000, axis=-1)
    groups = defaultdict(list)
    for j, index in enumerate(indices):
        row = rows[int(index)]
        target = (int(np.argmax(row['task'][4:9])), int(np.argmax(row['task'][2:4])))
        groups[('stage', int(actual_stage[j]))].append(j)
        groups[('target', *target)].append(j)
        groups[('episode', int(row['episode']))].append(j)
    def result(items):
        return dict(rows=len(items), stage_accuracy=float(np.mean(guessed_stage[items] == actual_stage[items])),
                    waypoint_translation_mm_median=float(np.median(mm[items])),
                    waypoint_translation_mm_p95=float(np.quantile(mm[items], .95)))
    return dict(overall=result(np.arange(len(indices))),
                by_group={str(key): result(np.asarray(value)) for key, value in sorted(groups.items())},
                predicted_stage_counts=dict(Counter(map(int, guessed_stage))),
                errors_mm=mm.tolist(),predicted_stages=guessed_stage.tolist())


def state_baselines(rows, train, validation):
    state = np.asarray([r['state'] + r['task'] for r in rows], float)
    xyz = np.asarray([r['subgoal'][:3] for r in rows], float)
    stage = np.asarray([r['stage'] for r in rows], int)
    mean = state[train].mean(0)
    std = np.maximum(state[train].std(0), 1e-4)
    fit = np.c_[(state[train] - mean) / std, np.ones(len(train))]
    held = np.c_[(state[validation] - mean) / std, np.ones(len(validation))]
    penalty = np.eye(fit.shape[1]);penalty[-1, -1] = 0
    solve = np.linalg.solve(fit.T @ fit + penalty, fit.T)
    stage_fit = solve @ stage[train]
    waypoint_fit = solve @ xyz[train]
    state_score = held @ stage_fit
    state_stage = (state_score >= .5).astype(int)
    state_xyz = held @ waypoint_fit
    majority = int(np.mean(stage[train]) >= .5)
    constant_stage = np.full(len(validation), majority, int)
    constant_xyz = np.tile(xyz[train].mean(0), (len(validation), 1))
    return dict(constant_mean=metrics(rows,validation,stage,xyz,
                                      np.eye(7)[constant_stage],constant_xyz),
                state_only_ridge=metrics(rows,validation,stage,xyz,
                                         np.eye(7)[state_stage],state_xyz),
                state_only_inputs='measured current 32D state plus public 10D task; no image/history',
                ridge_penalty=1.0)


@torch.inference_mode()
def evaluate(model, data, indices):
    model.eval()
    stage_logits=[];xyz=[]
    for part in np.array_split(indices, max(1,(len(indices)+127)//128)):
        if not len(part):continue
        ind=torch.as_tensor(part,device=data['features'].device)
        _,stage,subgoal,_=predict(model,data,ind,False,force_zero_previous=True)
        stage_logits.append(stage.float().cpu().numpy())
        xyz.append(subgoal[:,:3].float().cpu().numpy())
    labels=np.asarray([r['stage'] for r in data['rows']],int)
    waypoints=np.asarray([r['subgoal'] for r in data['rows']],float)
    return metrics(data['rows'],indices,labels,waypoints,
                   np.concatenate(stage_logits),np.concatenate(xyz))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--source-checkpoint',type=Path,required=True)
    parser.add_argument('--image-checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=2000)
    parser.add_argument('--budget-minutes',type=float,default=120)
    parser.add_argument('--seed',type=int,default=20261003)
    args=parser.parse_args()
    if args.output.exists():parser.error('Preserve previous probe output')
    if args.updates>2000 or args.budget_minutes>120:
        parser.error('Exceeded predeclared probe budget')
    args.output.mkdir(parents=True)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    data=build_data(args.cache,args.source_checkpoint,'cuda',LONG_AGES)
    data['prev'].zero_()  # Avoid teacher-command history as a privileged offline shortcut.
    rows=data['rows']
    train=np.asarray([i for i,r in enumerate(rows) if r['split']=='train'],np.int64)
    validation=np.asarray([i for i,r in enumerate(rows) if r['split']=='validation'],np.int64)
    if len({r['episode'] for r in rows if r['split']=='train'})!=50 or \
       len({r['episode'] for r in rows if r['split']=='validation'})!=10:
        parser.error('Expected 50/10 complete teacher episodes')
    stage_counts=Counter((rows[i]['episode'],rows[i]['stage']) for i in train)
    weights=np.asarray([1/(len(stage_counts)*stage_counts[(rows[i]['episode'],rows[i]['stage'])])
                        for i in train],float)
    weights/=weights.sum()
    cp=torch.load(args.image_checkpoint,map_location='cpu',weights_only=False)
    source_manifest=cp['source_manifest']
    if (not source_manifest['long_history'] or not source_manifest['full_state_skip'] or
        source_manifest['feature_dim']!=data['features'].shape[1]):
        parser.error('Frozen action-only initialization architecture mismatch')
    model=TemporalTransport(data['mean'],data['std'],raw_state_skip=True,
                            feature_dim=data['features'].shape[1],
                            state_task_dim=42,route_conditioning=True).cuda()
    model.load_state_dict(cp['model'],strict=True)
    for parameter in model.action.parameters():parameter.requires_grad_(False)
    for parameter in model.handoff.parameters():parameter.requires_grad_(False)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                lr=1e-4,weight_decay=1e-4)
    baselines=state_baselines(rows,train,validation)
    (args.output/'baselines.json').write_text(json.dumps(baselines,indent=2)+'\n')
    manifest=dict(schema='aic_sfp_route_stage_waypoint_probe/v1',
                  cache_manifest_sha256=sha(args.cache/'manifest.json'),
                  source_checkpoint_sha256=sha(args.source_checkpoint),
                  image_checkpoint_sha256=sha(args.image_checkpoint),
                  architecture='same 2-layer 256D temporal transport encoder; 7-way stage and 9D body-frame waypoint heads',
                  initialization='frozen ordinary full-action checkpoint; representation and route heads trainable; action/handoff heads frozen',
                  previous_teacher_commands_used=False,stage_and_waypoint_actor_input=False,
                  feature_inputs='frozen three-view ImageNet ResNet18 + causal image history + measured state/motion + public task',
                  train_episodes=50,validation_episodes=10,train_rows=len(train),validation_rows=len(validation),
                  updates_cap=args.updates,wall_minutes_cap=args.budget_minutes,
                  gate='>=90% held-out route-stage accuracy and <=20mm waypoint translation p95; grouped complete resets',
                  final_scenes_opened=False)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    best=float('inf');best_update=0;history=[];start=time.monotonic();stop='update_cap'
    for update in range(1,args.updates+1):
        if time.monotonic()-start>args.budget_minutes*60:
            stop='wall_budget';break
        chosen=np.random.choice(train,size=64,replace=True,p=weights)
        ind=torch.as_tensor(chosen,device='cuda')
        model.train();optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            _,stage,subgoal,_=predict(model,data,ind,False,force_zero_previous=True)
            stage_loss=nn.functional.cross_entropy(stage,data['stage'][ind])
            waypoint_loss=nn.functional.smooth_l1_loss(subgoal[:,:3]/.1,
                                                     data['subgoal'][ind,:3]/.1,beta=.1)
            rotation_loss=nn.functional.mse_loss(subgoal[:,3:],data['subgoal'][ind,3:])
            loss=stage_loss+waypoint_loss+.05*rotation_loss
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if update%250==0 or update==args.updates:
            outcome=evaluate(model,data,validation)
            overall=outcome['overall']
            score=overall['waypoint_translation_mm_p95']+20*(1-overall['stage_accuracy'])
            history.append(dict(update=update,score=score,overall=overall,loss=float(loss),
                                elapsed_min=(time.monotonic()-start)/60))
            print(json.dumps(history[-1]),flush=True)
            if score<best:
                best=score;best_update=update
                torch.save(dict(model=model.state_dict(),update=update,source_manifest=manifest,
                                route_conditioning=True),args.output/'best.pt')
                (args.output/'best_predictions.json').write_text(json.dumps(outcome)+'\n')
    best_outcome=json.loads((args.output/'best_predictions.json').read_text())
    result=dict(best_update=best_update,updates_completed=update,stop_reason=stop,
                validation=best_outcome,baselines={k:v['overall'] for k,v in baselines.items()
                                                   if isinstance(v,dict) and 'overall' in v},
                offline_gate_passed=(best_outcome['overall']['stage_accuracy']>=.90 and
                                     best_outcome['overall']['waypoint_translation_mm_p95']<=20),
                autonomous_gate_passed=False,final_scenes_opened=False)
    (args.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('validation','baselines')}),flush=True)


if __name__=='__main__':main()
