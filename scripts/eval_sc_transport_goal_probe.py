#!/usr/bin/env python3
"""Held-out geometry, false-arrival, and task-code checks for SC goal probe."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from train_sc_transport_goal_probe import GoalProbe, labels


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(8)
    rows=[json.loads(s) for s in (a.cache/'rows.jsonl').open()]
    feature=np.asarray(np.load(a.cache/'features.npy',mmap_mode='r'))
    base=torch.load(a.base,map_location='cpu',weights_only=False)
    state=np.asarray([r['state'] for r in rows],dtype=np.float32)
    state=(state-base['model']['state_mean'].numpy())/base['model']['state_std'].numpy()
    state[:,base.get('masked_state_indices',[])]=0
    task=np.asarray([r['task'] for r in rows],dtype=np.float32)
    indices=np.asarray([i for i,r in enumerate(rows) if r['family']=='sc' and r['split']=='validation'])
    x=np.concatenate((feature[indices],state[indices],task[indices]),axis=1).astype(np.float32)
    x_flip=x.copy();x_flip[:,-8:-6]=x[:,-8:-6][:,::-1]
    model=GoalProbe().cuda().eval()
    model.load_state_dict(torch.load(a.model,map_location='cpu',weights_only=False)['model'])
    def infer(values):
        parts=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for part in np.array_split(values,max(1,(len(values)+511)//512)):
                parts.append(model(torch.as_tensor(part,device='cuda')).float().cpu().numpy())
        return np.concatenate(parts)*np.array([.2,.2,.2,.3,.3,.3])
    pred=infer(x);flipped=infer(x_flip);truth=labels(rows)[indices]
    err=np.linalg.norm(pred[:,:3]-truth[:,:3],axis=1)*1000
    orient_err=np.linalg.norm(pred[:,3:]-truth[:,3:],axis=1)*180/np.pi
    true_distance=np.linalg.norm(truth[:,:3],axis=1)*1000
    pred_distance=np.linalg.norm(pred[:,:3],axis=1)*1000
    true_rotation=np.linalg.norm(truth[:,3:],axis=1)*180/np.pi
    pred_rotation=np.linalg.norm(pred[:,3:],axis=1)*180/np.pi
    by_distance={}
    for name,mask in [('near10',true_distance<=10),('near20',true_distance<=20),
                      ('near50',true_distance<=50),('far50plus',true_distance>50)]:
        by_distance[name]={'rows':int(mask.sum()),
            'translation_median_mm':float(np.median(err[mask])) if mask.any() else None,
            'translation_p95_mm':float(np.quantile(err[mask],.95)) if mask.any() else None,
            'orientation_p95_deg':float(np.quantile(orient_err[mask],.95)) if mask.any() else None}
    events=[]
    for threshold in (4,6,8,10,12,15,20):
        for streak in (1,2,3):
            groups=defaultdict(list)
            for j,i in enumerate(indices):
                groups[rows[int(i)]['episode']].append((rows[int(i)]['time_s'],j))
            early=correct=late=missed=0
            for seq in groups.values():
                seq.sort()
                valid=[t for t,j in seq if true_distance[j]<=10 and true_rotation[j]<=3]
                count=0;trigger=None
                for t,j in seq:
                    count=(count+1 if pred_distance[j]<=threshold and pred_rotation[j]<=3 else 0)
                    if count>=streak:
                        trigger=t;break
                if trigger is None:missed+=1
                elif not valid or trigger<min(valid):early+=1
                elif trigger<=max(valid):correct+=1
                else:late+=1
            accepted=(pred_distance<=threshold)&(pred_rotation<=3)
            truth_accept=(true_distance<=10)&(true_rotation<=3)
            events.append({'threshold_mm':threshold,'streak':streak,
                'early_episodes':early,'correct_episodes':correct,'late_episodes':late,
                'missed_episodes':missed,'accepted_rows':int(accepted.sum()),
                'accepted_true_10mm_3deg_fraction':float(truth_accept[accepted].mean()) if accepted.any() else None})
    stage5=np.asarray([rows[int(i)]['stage']==5 for i in indices])
    flip_shift=np.linalg.norm(pred[:,:3]-flipped[:,:3],axis=1)*1000
    servo=[]
    groups=defaultdict(list)
    for j,i in enumerate(indices):
        groups[rows[int(i)]['episode']].append((rows[int(i)]['time_s'],j))
    for threshold in (25,30,35,40,45,50):
        for streak in (1,2,3):
            early=correct=missed=0;trigger_stages=[]
            for seq in groups.values():
                count=0;trigger=None
                for _,j in sorted(seq):
                    count=(count+1 if pred_distance[j]<=threshold and pred_rotation[j]<=3 else 0)
                    if count>=streak:trigger=rows[int(indices[j])]['stage'];break
                if trigger is None:missed+=1
                elif trigger<5:early+=1;trigger_stages.append(int(trigger))
                else:correct+=1;trigger_stages.append(int(trigger))
            servo.append({'threshold_mm':threshold,'streak':streak,
                'early_stage_episodes':early,'stage5_episodes':correct,'missed_episodes':missed,
                'trigger_stages':trigger_stages})
    report={'schema':'aic_sc_transport_goal_probe_eval/v1','checkpoint':str(a.model),
        'validation_episodes':len({rows[int(i)]['episode'] for i in indices}),
        'validation_rows':len(indices),'error_by_true_distance':by_distance,
        'selected_port_flip_shift_stage5_mm':{'median':float(np.median(flip_shift[stage5])),
            'p95':float(np.quantile(flip_shift[stage5],.95))},
        'arrival_calibration':events,'servo_activation':servo,'final_scenes_opened':False}
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'distance':by_distance,'port_flip':report['selected_port_flip_shift_stage5_mm'],
        'best_event_checks':sorted(events,key=lambda v:(v['early_episodes'],-v['correct_episodes']))[:6],
        'servo':servo},indent=2))


if __name__=='__main__':main()
