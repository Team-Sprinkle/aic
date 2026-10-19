#!/usr/bin/env python3
"""Collect bounded train-only same-board alternate-request SFP teacher controls.

The original 50 scored teacher rolls provide scene/reset identities. This
script changes only the public requested target, never uses source actions,
and stops if the five-run feasibility gate fails.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import yaml

from run_sfp_cheatcode_target_pilot import BASE, ROOT, TEMPLATE, score


SOURCE=Path('/home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/balanced_teacher_train/progress.json')
OLD_PLAN=ROOT/'outputs/experiments/2026-10-05_transport_identity_audit/paired_request_plan_v2/plan.json'


def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()


def plan(output):
    source=json.loads(SOURCE.read_text())['results']
    used={Path(r['source_trial']).stem for r in json.loads(OLD_PLAN.read_text())['pairs']}
    groups=defaultdict(list)
    for item in source.values():
        if item['official_score'].get('tier3',0)<75:continue
        trial=ROOT/item['source_trial']
        if trial.stem in used:continue
        groups[(item['card'],item['port'])].append(item)
    if len(groups)!=10:raise ValueError(f'Expected ten original requests, got {len(groups)}')
    chosen=[]
    for key,items in sorted(groups.items()):
        items.sort(key=lambda r:(r['card_count'],r['source_trial']))
        if len(items)<2:raise ValueError(f'Missing two train scenes for {key}')
        chosen.extend((items[0],items[-1]))
    rows=[]
    for item in chosen:
        src=ROOT/item['source_trial'];cfg=yaml.safe_load(src.read_text())
        trial=next(iter(cfg['trials'].values()))
        scene_digest=digest({'scene':trial['scene'],'robot':cfg.get('robot')})
        board=trial['scene']['task_board']
        present=[i for i in range(5) if board[f'nic_rail_{i}']['entity_present']]
        card=item['card'];port=item['port']
        alt_card=present[(present.index(card)+1)%len(present)] if len(present)>1 else card
        alt_port=1-port
        task=next(iter(trial['tasks'].values()))
        task['target_module_name']=f'nic_card_mount_{alt_card}'
        task['port_name']=f'sfp_port_{alt_port}'
        if digest({'scene':trial['scene'],'robot':cfg.get('robot')})!=scene_digest:
            raise AssertionError('Scene/reset changed')
        name=f'card{card}_port{port}_{src.stem}_to_card{alt_card}_port{alt_port}'
        folder=output/'scenes'/name;folder.mkdir(parents=True,exist_ok=True)
        (folder/'eval_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
        for file in ('world_audit.sdf','bridge_audit.yaml'):
            shutil.copy2(TEMPLATE/file,folder/file)
        rows.append(dict(name=name,source_trial=str(src),original_request=[card,port],
                         alternate_request=[alt_card,alt_port],card_count=item['card_count'],
                         scene_digest=scene_digest,scene_dir=str(folder),
                         config_sha256=hashlib.sha256((folder/'eval_config.yaml').read_bytes()).hexdigest(),
                         original_score=item['official_score']))
    return dict(schema='aic_sfp_paired_coverage_v2',train_only=True,
                final_scenes_opened=False,source_agent_actions_used=False,
                gate='first five >=4 full insertions; do not admit unsuccessful action labels',
                scenes=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',default='1')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    plan_path=a.output/'plan.json'
    if plan_path.exists():planned=json.loads(plan_path.read_text())
    else:
        planned=plan(a.output)
        plan_path.write_text(json.dumps(planned,indent=2)+'\n')
    if (planned['schema']!='aic_sfp_paired_coverage_v2' or
            (not planned['train_only'] and planned.get('source_split')!='development') or
            planned['final_scenes_opened']):
        raise ValueError('Unexpected plan')
    progress_path=a.output/'progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'results':{}}
    for index,spec in enumerate(planned['scenes']):
        if index==5:
            first=[progress['results'].get(x['name'],{}).get('official_score',{}).get('tier3',0)
                   for x in planned['scenes'][:5]]
            if sum(x>=75 for x in first)<4:
                print(json.dumps(dict(stopped='five_run_gate',scores=first)),flush=True)
                break
        name=spec['name']
        if name in progress['results']:continue
        scene=Path(spec['scene_dir'])
        if hashlib.sha256((scene/'eval_config.yaml').read_bytes()).hexdigest()!=spec['config_sha256']:
            raise ValueError('Changed config')
        audit=a.output/'rollouts'/name
        env=os.environ.copy();env.update(DOCKER_HOST=f'unix:///run/user/{os.getuid()}/docker.sock',
            AIC_TRANSPORT_EVAL_GPU=a.gpu,AIC_TRANSPORT_EVAL_POLICY_GPU=a.gpu,
            AIC_TRANSPORT_EVAL_SCENE_DIR=str(scene.resolve()),
            AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA='.06',
            AIC_TRANSPORT_EVAL_POLICY='aic_example_policies.ros.CheatCode',
            AIC_TRANSPORT_EVAL_TEACHER_PREFIX='true',
            AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC='70',
            AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC='200')
        cmd=['bash','scripts/run_clean_transport_eval.sh',name,str(BASE),str(audit)]
        with (a.output/(name+'.launch.log')).open('w') as log:
            try:run=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                                   timeout=350)
            except subprocess.TimeoutExpired:run=None
        result=dict(returncode=run.returncode if run else None,official_score=score(audit),
                    audit=str(audit),command=cmd)
        progress['results'][name]=result
        progress_path.write_text(json.dumps(progress,indent=2)+'\n')
        print(json.dumps(dict(name=name,**result)),flush=True)
        if run is None or run.returncode or not result['official_score']['available']:
            print(json.dumps(dict(stopped='run_error',name=name)),flush=True);break
    print(json.dumps(dict(completed=len(progress['results']),
        full=sum(r['official_score'].get('tier3',0)>=75 for r in progress['results'].values()))),flush=True)


if __name__=='__main__':main()
