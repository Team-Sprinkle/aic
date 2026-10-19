#!/usr/bin/env python3
"""Offline SC route-rejoin actions for already recorded actor observations.

The MoveIt path is an offline training label only. Candidate matching is
monotone and bounded so a crossing segment cannot silently become the target.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from train_shared_port_tcp_bc import relative_tcp_target9


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing output')
    a.output.mkdir(parents=True)
    rows=[json.loads(s) for s in (a.cache/'rows.jsonl').open()]
    source={e['case']:e for e in json.loads(a.dataset.read_text())['episodes']}
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[row['case']].append(i)
    targets=np.zeros((len(rows),4,9),dtype=np.float32)
    valid=np.zeros(len(rows),dtype=bool)
    matching=[]
    for case,indices in sorted(groups.items()):
        e=source[case]
        plan_file=Path(e['insert_plan'])/'moveit_joint_trajectory.json'
        plan=json.loads(plan_file.read_text())['waypoints']
        xyz=np.asarray([w['tcp_pose']['position'] for w in plan],dtype=float)
        plan_times=np.asarray([w['timestamp'] for w in plan],dtype=float)
        poses=[np.r_[w['tcp_pose']['position'],w['tcp_pose']['orientation_xyzw']] for w in plan]
        indices.sort(key=lambda i:rows[i]['sim_time'])
        actor_times=np.asarray([rows[i]['sim_time'] for i in indices],dtype=float)
        actor_positions=np.asarray([rows[i]['state'][:3] for i in indices],dtype=float)
        goal_dist=np.linalg.norm(actor_positions-xyz[-1],axis=1)
        nearest_goal_row=int(goal_dist.argmin())
        actor_goal_time=max(1.,actor_times[nearest_goal_row]-actor_times[0])
        matched=0;errors=[];accepted=0;last_index=[]
        for i in indices:
            row=rows[i];state=np.asarray(row['state'][:7],dtype=float)
            t=float(row['sim_time'])
            # Offline time alignment resolves path crossings. The actor's
            # closest approach time is a training label only, never input.
            nominal=np.clip((t-actor_times[0])/actor_goal_time,0,1)*plan_times[-1]
            candidates=np.flatnonzero((plan_times>=nominal-3.) &
                                       (plan_times<=nominal+3.) &
                                       (np.arange(len(plan))>=matched))
            if not len(candidates):candidates=np.arange(matched,len(plan))
            distances=np.linalg.norm(xyz[candidates]-state[:3],axis=1)
            score=distances+abs(plan_times[candidates]-nominal)*.001
            matched=int(candidates[np.argmin(score)])
            error=float(np.linalg.norm(xyz[matched]-state[:3]))
            errors.append(error*1000);last_index.append(matched)
            if error>.08:continue
            for h,advance in enumerate((2,4,6,8)):
                target=poses[min(len(poses)-1,matched+advance)]
                targets[i,h]=relative_tcp_target9(state,target)
            valid[i]=True;accepted+=1
        matching.append({'case':case,'split':rows[indices[0]]['split'],
            'scene_reset_group':e['scene_reset_group'],'rows':len(indices),
            'valid_under_80mm':accepted,'match_error_p50_mm':float(np.median(errors)),
            'match_error_p95_mm':float(np.quantile(errors,.95)),
            'final_matched_waypoint':last_index[-1],
            'plan_waypoints':len(plan),'actor_nearest_goal_time_s':float(actor_goal_time),
            'teacher_plan':str(plan_file)})
    np.save(a.output/'targets.npy',targets)
    np.save(a.output/'valid.npy',valid)
    manifest={'schema':'aic_sc_onpolicy_route_correction_labels/v1',
        'source_cache':str(a.cache/'manifest.json'),'matching':matching,
        'target':'four complete TCP-body targets at +0.2/+0.4/+0.6/+0.8s plan lookahead; offline monotone time alignment',
        'exclusion':'matched TCP farther than 80 mm from route',
        'teacher_plan_training_only':True,'final_scenes_opened':False}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(matching,indent=2))


if __name__=='__main__':main()
