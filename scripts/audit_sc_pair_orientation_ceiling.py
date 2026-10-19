#!/usr/bin/env python3
"""Posthoc ceiling for orientation inferred only from the SC port-pair axis.

This intentionally uses scored geometry, so it is a diagnostic upper bound,
never a deployable orientation estimate or an evaluation crop selector.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def stats(values):
    return {'episodes': len(values), 'median_deg': float(np.median(values)),
            'p95_deg': float(np.quantile(values, .95)),
            'over_3deg': int(np.sum(np.asarray(values)>3))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--pairs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous result')
    episodes=json.loads(a.dataset.read_text())['episodes']
    pairs={r['case']:r for r in json.loads(a.pairs.read_text())['episodes'].values()}
    rows=[]
    for e in episodes:
        case=e['case'];pair=np.asarray(pairs[case]['port0_minus_port1_standoff_base_m'])
        plan=json.loads((Path(e['insert_plan'])/'moveit_joint_trajectory.json').read_text())
        goal=Rotation.from_quat(plan['waypoints'][-1]['tcp_pose']['orientation_xyzw'])
        rows.append({'case':case,'split':e['split'],'pair_base_m':pair,'goal':goal})
    train=[r for r in rows if r['split']=='train']
    if {r['case'] for r in train}&{r['case'] for r in rows if r['split']=='validation'}:
        raise ValueError('Episode split leakage')
    prior=Rotation.from_quat([r['goal'].as_quat() for r in train]).mean()
    reference=np.median([r['goal'].inv().apply(r['pair_base_m']) for r in train],axis=0)
    results=[]
    for r in rows:
        predicted_pair=prior.apply(reference)
        delta=np.arctan2(r['pair_base_m'][1],r['pair_base_m'][0])-\
              np.arctan2(predicted_pair[1],predicted_pair[0])
        estimated=Rotation.from_euler('z',delta)*prior
        results.append({'case':r['case'],'split':r['split'],
            'pair_axis_oracle_error_deg':float(np.rad2deg((estimated.inv()*r['goal']).magnitude())),
            'train_mean_error_deg':float(np.rad2deg((prior.inv()*r['goal']).magnitude()))})
    metrics={}
    for split in ('train','validation'):
        subset=[r for r in results if r['split']==split]
        metrics[split]={'oracle_pair_axis':stats([r['pair_axis_oracle_error_deg'] for r in subset]),
                        'train_mean':stats([r['train_mean_error_deg'] for r in subset])}
    output={'schema':'aic_sc_oracle_port_pair_axis_orientation_ceiling/v1',
        'meaning':'Posthoc scored pair-axis upper bound; not a deployable model',
        'source_dataset':str(a.dataset),'source_pairs':str(a.pairs),
        'train_only_prior_quat_xyzw':prior.as_quat().tolist(),
        'train_only_pair_vector_in_goal_frame_m':reference.tolist(),
        'metrics':metrics,'episodes':results,'geometry_runtime_input':True,
        'final_scenes_opened':False}
    a.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
