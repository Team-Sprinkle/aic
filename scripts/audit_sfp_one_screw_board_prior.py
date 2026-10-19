#!/usr/bin/env python3
"""Privileged one-visual-screw upper bound for SFP requested-port association."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import yaml

from audit_sc_port_targets import matrix
from build_sfp_port_tcp_waypoints import transform


SCREW=np.asarray((0.145,-0.2075,0.010))


def key(r):return r['scene_group'],r['episode'],round(float(r['time_s']),6)


def xyz_target(row,identity):
    return np.asarray(next(t for t in row['targets'] if t['identity']==identity)['position_base_m'],float)


def summary(rows):
    errors=np.asarray([x['error_mm'] for x in rows])
    return {'groups':len(rows),'median_mm':float(np.median(errors)),
            'p95_mm':float(np.percentile(errors,95)),
            'within_10mm_groups':int(sum(errors<=10)),
            'wrong_installed_opening_groups':int(sum(x['wrong_installed_opening'] for x in rows))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rail-rows',type=Path,required=True)
    p.add_argument('--source-rows',type=Path,required=True)
    p.add_argument('--sfp-calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous output')
    a.output.mkdir(parents=True)
    rails=[json.loads(l) for l in a.rail_rows.open()]
    sources={key(r):r for r in (json.loads(l) for l in a.source_rows.open())}
    sfp=json.loads(a.sfp_calibration.read_text())
    base_world=np.linalg.inv(matrix(sfp['world_to_base']))
    groups={}
    for r in rails:
        if r['autonomous_actor_diagnostic']:continue
        if r['scene_group'] in groups:continue
        source=sources[key(r)]
        if source['images']!=r['images']:raise ValueError('Image provenance mismatch')
        trial=next(iter(yaml.safe_load(Path(r['trial_yaml_label_only']).read_text())['trials'].values()))
        bp=trial['scene']['task_board']['pose']
        board_base=base_world@transform([bp[k] for k in ('x','y','z')],
                                        [bp[k] for k in ('roll','pitch','yaw')])
        screw=board_base[:3,:3]@SCREW+board_base[:3,3]
        target=xyz_target(source,r['selected_identity'])
        local=board_base[:3,:3].T@(target-board_base[:3,3])
        groups[r['scene_group']]={'split':r['split'],'identity':r['selected_identity'],
            'card_count':r['card_count'],'occupied':r['occupied_slots_training_label_only'],
            'yaw':float(bp['yaw']),'board_rotation':board_base[:3,:3],
            'screw':screw,'target':target,'target_local':local,'source':source}
    train=[x for x in groups.values() if x['split']=='train']
    dev=[x for x in groups.values() if x['split']=='validation']
    if len(train)!=70 or len(dev)!=10:raise ValueError(f'Grouped split {len(train)}/{len(dev)}')
    yaw=float(np.arctan2(np.mean(np.sin([x['yaw'] for x in train])),
                         np.mean(np.cos([x['yaw'] for x in train]))))
    c,s=np.cos(yaw),np.sin(yaw)
    mean_rotation=base_world[:3,:3]@np.array(((c,-s,0),(s,c,0),(0,0,1)))
    local_by_identity=defaultdict(list)
    for x in train:local_by_identity[x['identity']].append(x['target_local'])
    if len(local_by_identity)!=10:raise ValueError('Missing requested identity')
    local_prior={name:np.median(np.asarray(v),axis=0) for name,v in local_by_identity.items()}
    screw_mean=np.mean([x['screw'] for x in train],axis=0)
    variants={'oracle_screw_train_yaw':[],'oracle_screw_true_yaw':[],
              'mean_screw_train_yaw':[]}
    for scene,x in groups.items():
        if x['split']!='validation':continue
        delta=local_prior[x['identity']]-SCREW
        for variant in variants:
            screw=x['screw'] if not variant.startswith('mean_') else screw_mean
            rotation=x['board_rotation'] if variant.endswith('true_yaw') else mean_rotation
            pred=screw+rotation@delta
            distances=[]
            for t in x['source']['targets']:
                card=int(t['identity'].split('_')[2])
                if not x['occupied'][card]:continue
                distances.append((float(np.linalg.norm(pred-t['position_base_m'])),t['identity']))
            nearest=min(distances)[1]
            variants[variant].append({'scene_group':scene,'identity':x['identity'],
                'card_count':x['card_count'],'error_mm':float(np.linalg.norm(pred-x['target'])*1000),
                'nearest_installed_opening':nearest,
                'wrong_installed_opening':nearest!=x['identity'],
                'predicted_port_base_m':pred.tolist()})
    result={'schema':'sfp_one_mesh_screw_board_prior/v1','train_groups':70,
            'development_groups':10,'mean_train_board_yaw_rad':yaw,
            'screw_local_board_m':SCREW.tolist(),
            'local_offsets_board_m':{k:v.tolist() for k,v in local_prior.items()},
            'variants':{name:summary(rows) for name,rows in variants.items()},
            'gate_passed':summary(variants['oracle_screw_train_yaw'])['within_10mm_groups']>=9
              and summary(variants['oracle_screw_train_yaw'])['wrong_installed_opening_groups']==0,
            'true_screw_used_for_diagnostic_only':True,'final_scenes_opened':False}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,rows in variants.items():
        (a.output/f'{name}.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(result['variants'],indent=2))


if __name__=='__main__':main()
