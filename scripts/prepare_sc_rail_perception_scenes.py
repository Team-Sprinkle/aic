#!/usr/bin/env python3
"""New nonfinal perception-only resets covering the released SC rail range.

Original SC LLB experts cover only a narrow rail layout. These recordings may
fail insertion and are never admitted as successful BC demonstrations.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import numpy as np
import yaml


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    template=yaml.safe_load((a.source/'eval_config.yaml').read_text());xacro=(a.source/'ur_llb_start.xacro').read_text()
    rng=np.random.default_rng(2026101541);spec=[]
    for n in range(1,6):
        for port in (0,1):
            for rep in (0,1):spec.append(('train',n,port,rep,None))
        spec.append(('validation',n,n%2,2,None))
    spec.extend([(split,n,port,3,1-port) for split,n in [('train',1),('validation',5)] for port in (0,1)])
    cases=[];bins=np.linspace(-.06,.055,5)
    for i,(split,n,port,rep,absent) in enumerate(spec):
        name=f'{split}_n{n}_p{port}_r{rep}'+(f'_absent{absent}' if absent is not None else '')
        scene=a.output/'scenes'/name;scene.mkdir(parents=True)
        cfg=copy.deepcopy(template);trial=next(iter(cfg['trials'].values()));cfg['trials']={f'trial_{1950000+i}':trial}
        b=trial['scene']['task_board'];b['pose'].update(x=float(rng.uniform(.12,.18)),y=float(rng.uniform(-.24,-.12)),yaw=float(rng.uniform(2.95,3.20)),roll=0.,pitch=0.)
        selected=sorted(rng.choice(5,n,replace=False).tolist())
        for card in range(5):
            rail=b[f'nic_rail_{card}'];rail.update(entity_present=card in selected,entity_name=f'nic_card_{card}')
            rail['entity_pose']=dict(translation=float(rng.uniform(-.0215,.0234)),roll=0.,pitch=0.,yaw=float(rng.uniform(-np.pi/18,np.pi/18)))
        for q in (0,1):
            rail=b[f'sc_rail_{q}'];rail.update(entity_present=q!=absent,entity_name=f'sc_mount_{q}')
            idx=(n+2*rep+port)%5 if q==0 else (2*n+rep+3*port)%5
            translation=(float(np.clip(bins[idx]+rng.uniform(-.003,.003),-.06,.055)) if split=='train' else float(rng.uniform(-.056,.051)))
            rail['entity_pose']=dict(translation=translation,roll=0.,pitch=0.,yaw=0.)
        task=next(iter(trial['tasks'].values()));task.update(target_module_name=f'sc_port_{port}',time_limit=180)
        joints={k:float(v+rng.uniform(-.006,.006)) for k,v in cfg['robot']['home_joint_positions'].items()}
        cfg['robot']['home_joint_positions']=joints;xml=xacro
        for joint,value in joints.items():
            xml,count=re.subn(rf'({re.escape(joint)}=)(-?[0-9]+\.[0-9]+)',rf'\g<1>{value:.7f}',xml,count=1)
            if count!=1:raise ValueError(f'Cannot set actual robot start for{joint}')
        (scene/'ur_llb_start.xacro').write_text(xml)
        (scene/'eval_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
        for f in ('world_audit.sdf','bridge_audit.yaml'):shutil.copy2(a.source/f,scene/f)
        physical=hashlib.sha256(json.dumps(dict(scene=trial['scene'],robot=cfg['robot']),sort_keys=True).encode()).hexdigest()
        cases.append(dict(case=name,scene=str(scene),family='sc',split=split,card_count=n,port=port,requested_card=None,present_cards=selected,replicate=rep,domain='new_SC_rail_coverage',physical_reset_group=physical,SC_rail_translation_m=[b[f'sc_rail_{q}']['entity_pose']['translation'] for q in (0,1)],absent_port=absent,perception_only=True,BC_eligible=False,configured_start_joints=joints,final_scene=False))
    assert len({c['physical_reset_group'] for c in cases})==len(cases)
    report=dict(schema='SC_rail_perception_collection_plan/v1',cases=cases,seed=2026101541,
        hypothesis='SC0/1 training rail ranges[-1.8,18]mm/[-60,-43]mm exclude released evaluation settings. Cover independent rail translations and counterfactual wrong/absent requests with native RGB.',
        first_gate='First2recordings: native image shape, causal state/TF joins and actual green GT pixels visually checked before continuing.',
        stopping_rule='29recordings maximum; stop on infrastructure/provenance/label failure. No insertion or BC success claims.',
        source=str(a.source),final_scenes_opened=False)
    (a.output/'plan.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(cases=len(cases),train=sum(c['split']=='train' for c in cases),validation=sum(c['split']=='validation' for c in cases))))


if __name__=='__main__':main()
