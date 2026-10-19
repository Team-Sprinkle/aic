#!/usr/bin/env python3
"""Posthoc threshold check for an observation-only SC transport stop signal.

Recorded teacher standoffs are used only to score trigger quality. Complete
scene/reset groups remain separated by the cache manifest.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_sc_transport_goal_probe import GoalProbe


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threshold-mm',type=float,default=30.)
    p.add_argument('--streak',type=int,default=3)
    a=p.parse_args()
    torch.set_num_threads(8)
    rows=[json.loads(s) for s in (a.cache/'rows.jsonl').open()]
    features=np.asarray(np.load(a.cache/'features.npy',mmap_mode='r'),dtype=np.float32)
    saved=torch.load(a.base,map_location='cpu',weights_only=False)
    state=np.asarray([r['state'] for r in rows],dtype=np.float32)
    state=(state-saved['model']['state_mean'].numpy())/saved['model']['state_std'].numpy()
    state[:,saved.get('masked_state_indices',[])]=0
    task=np.asarray([r['task'] for r in rows],dtype=np.float32)
    x=np.concatenate((features,state,task),axis=1)
    goal=GoalProbe().eval()
    goal.load_state_dict(torch.load(a.model,map_location='cpu',weights_only=False)['model'])
    with torch.inference_mode():
        predicted=np.concatenate([goal(torch.as_tensor(chunk)).numpy()
                                  for chunk in np.array_split(x,max(1,(len(x)+255)//256))])
    predicted*=np.asarray([.2,.2,.2,.3,.3,.3],dtype=np.float32)
    cases={r['case'] for r in rows}
    results=[]
    for case in sorted(cases):
        ids=sorted((i for i,r in enumerate(rows) if r['case']==case),
                   key=lambda i:rows[i]['sim_time'])
        truth=np.asarray([rows[i]['goal_relative'] for i in ids])
        true_mm=np.linalg.norm(truth[:,:3],axis=1)*1000
        predicted_mm=np.linalg.norm(predicted[ids,:3],axis=1)*1000
        predicted_deg=np.linalg.norm(predicted[ids,3:],axis=1)*180/np.pi
        count=0;first=None
        for j,(distance,angle) in enumerate(zip(predicted_mm,predicted_deg)):
            count=count+1 if distance<=a.threshold_mm and angle<=5 else 0
            if count>=a.streak:
                first=j;break
        row=rows[ids[0]]
        results.append({'case':case,'split':row['split'],
                        'scene_reset_group':row['scene_reset_group'],
                        'card_count':row['card_count'],'selected_port':row['selected_port'],
                        'rows':len(ids),'true_nearest_mm':float(true_mm.min()),
                        'first_trigger_true_distance_mm':float(true_mm[first]) if first is not None else None,
                        'first_trigger_predicted_distance_mm':float(predicted_mm[first]) if first is not None else None,
                        'first_trigger_sim_time_s':float(rows[ids[first]]['sim_time']) if first is not None else None})
    report={'schema':'aic_sc_goal_stop_onpolicy_posthoc/v1',
            'cache_manifest':str(a.cache/'manifest.json'),'probe_checkpoint':str(a.model),
            'threshold_mm':a.threshold_mm,'streak':a.streak,'max_predicted_angle_deg':5,
            'results':results,'final_scenes_opened':False,
            'teacher_goal':'posthoc scoring only; never runtime input'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__': main()
