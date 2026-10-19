#!/usr/bin/env python3
"""Bounded train-only SFP coverage collection from historical scene configs.

Historical agent actions are never used. Each selected config runs a fresh
ordinary CheatCode Gazebo teacher and must pass its official insertion score.
"""
import argparse
from collections import Counter
import functools
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import yaml

from run_sfp_cheatcode_target_pilot import BASE, ROOT, TEMPLATE, score


SOURCE_ROOT=ROOT/'outputs/trajectory_datasets/clean_including_no_insert_trajs/sfp_to_nic'
PATTERNS=(
    'agent/nic_cards_mixed/n400__ec2_02_sfp_hard/trials/trial_*.yaml',
    'agent/nic_cards_mixed/n400__ec2_01_sfp_easy/trials/trial_*.yaml',
    'cheatcode/nic_cards_mixed/n200__card2_port0/trials/trial_*.yaml',
    'cheatcode/nic_cards_mixed/n200__card0_port1/trials/trial_*.yaml',
)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


@functools.lru_cache(None)
def config(path):
    return yaml.safe_load(Path(path).read_text())


def board_count(board):
    return sum(bool(board[f'nic_rail_{i}']['entity_present']) for i in range(5))


def plan(labels,output,min_per_cell,exclude_plans=()):
    episodes={}
    for line in labels.open():
        r=json.loads(line);episodes.setdefault(int(r['episode_index']),r)
    existing=Counter();new_development_validation=Counter();blocked=set()
    for r in episodes.values():
        blocked.add(r['scene_sha256'])
        task=r['task_vector'];card=max(range(5),key=lambda i:task[4+i])
        port=max(range(2),key=lambda i:task[2+i])
        if 'source_scene_yaml' in r:
            cfg=config(r['source_scene_yaml'])
            board=next(iter(cfg['trials'].values()))['scene']['task_board']
        else:
            stream=json.loads(Path(r['source_scored_stream']).read_text())
            board=stream['reset_identity']['scene']['task_board']
        key=(board_count(board),card,port)
        if r['split']=='train':existing[key]+=1
        elif r['split']=='validation' and int(r['episode_index'])>=2400000:
            new_development_validation[key]+=1
    for prior in exclude_plans:
        for spec in json.loads((prior/'plan.json').read_text())['scenes']:
            blocked.add(spec['scene_digest'])
    candidates={}
    for pattern in PATTERNS:
        for path in SOURCE_ROOT.glob(pattern):
            cfg=config(str(path));trial=next(iter(cfg['trials'].values()))
            board=trial['scene']['task_board'];task=next(iter(trial['tasks'].values()))
            card=int(task['target_module_name'].rsplit('_',1)[-1])
            port=int(task['port_name'].rsplit('_',1)[-1])
            count=board_count(board)
            if not board[f'nic_rail_{card}']['entity_present']:
                raise ValueError(f'Requested card absent in {path}')
            scene=digest({'scene':trial['scene'],'robot':cfg.get('robot')})
            if scene in blocked:continue
            key=(count,card,port)
            candidates.setdefault(key,[]).append((path,scene))
    chosen=[];used=set()
    for count in range(1,6):
        for card in range(5):
            for port in range(2):
                key=(count,card,port)
                need=max(0,min_per_cell-existing[key]-new_development_validation[key])
                if need==0:continue
                available=sorted(candidates.get(key,[]),
                    key=lambda x:hashlib.sha256(str(x[0]).encode()).hexdigest())
                picked=0
                for path,scene in available:
                    if scene in used:continue
                    used.add(scene);picked+=1
                    name=f'cards{count}_card{card}_port{port}_{path.parent.parent.name}_{path.stem}'
                    folder=output/'scenes'/name;folder.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(path,folder/'eval_config.yaml')
                    for item in ('world_audit.sdf','bridge_audit.yaml'):
                        shutil.copy2(TEMPLATE/item,folder/item)
                    chosen.append(dict(name=name,source_trial=str(path),card_count=count,
                                       request=[card,port],scene_digest=scene,scene_dir=str(folder),
                                       config_sha256=hashlib.sha256((folder/'eval_config.yaml').read_bytes()).hexdigest()))
                    if picked==need:break
                if picked<need:raise ValueError(f'Only {picked}/{need} new scenes for {key}')
    # Put one sample from each sparse request first, then round-robin other
    # cells; the first 20 rollouts test uncommon target/card combinations.
    by_key={}
    for row in chosen:by_key.setdefault((row['card_count'],*row['request']),[]).append(row)
    ordered=[]
    for count in (1,2,3,4,5):
        for card in range(5):
            for port in (1,0):
                key=(count,card,port)
                if by_key.get(key):ordered.append(by_key[key].pop(0))
    while any(by_key.values()):
        for key in sorted(by_key):
            if by_key[key]:ordered.append(by_key[key].pop(0))
    return dict(schema='aic_sfp_diverse_teacher/v1',train_only=True,
                final_scenes_opened=False,source_agent_actions_used=False,
                source_labels=str(labels),target_min_success_per_cell=min_per_cell,
                excluded_collection_plans=[str(x) for x in exclude_plans],
                current_train_counts={str(k):v for k,v in sorted(existing.items())},
                heldout_new_development_counts={str(k):v for k,v in sorted(new_development_validation.items())},
                pilot_count=20,pilot_min_full=16,rolling_30_min_full=23,
                scenes=ordered)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',default='1')
    p.add_argument('--min-per-cell',type=int,default=5)
    p.add_argument('--exclude-plan',type=Path,action='append',default=[],
                   help='Exclude every scene in a prior collection, including its failed runs')
    p.add_argument('--plan-only',action='store_true')
    a=p.parse_args()
    if a.gpu not in tuple(str(i) for i in range(8)):p.error('GPU must be 0..7')
    if not 2<=a.min_per_cell<=8:p.error('Minimum per cell must be 2..8')
    a.output.mkdir(parents=True,exist_ok=True)
    plan_path=a.output/'plan.json'
    if plan_path.exists():planned=json.loads(plan_path.read_text())
    else:
        planned=plan(a.labels,a.output,a.min_per_cell,a.exclude_plan)
        plan_path.write_text(json.dumps(planned,indent=2)+'\n')
    if planned['schema']!='aic_sfp_diverse_teacher/v1' or not planned['train_only'] or planned['final_scenes_opened']:
        raise ValueError('Invalid immutable plan')
    print(json.dumps(dict(planned=len(planned['scenes']),pilot=planned['pilot_count'])),flush=True)
    if a.plan_only:return
    progress_path=a.output/'progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'results':{}}
    for number,spec in enumerate(planned['scenes']):
        if number==planned['pilot_count']:
            scores=[progress['results'][x['name']]['official_score'].get('tier3',0)
                    for x in planned['scenes'][:number]]
            if sum(x>=75 for x in scores)<planned['pilot_min_full']:
                print(json.dumps(dict(stopped='pilot_gate',full=sum(x>=75 for x in scores))),flush=True)
                break
        if number>=50 and number%30==20:
            block=planned['scenes'][number-30:number]
            full=sum(progress['results'][x['name']]['official_score'].get('tier3',0)>=75 for x in block)
            if full<planned['rolling_30_min_full']:
                print(json.dumps(dict(stopped='rolling_success_gate',last30_full=full)),flush=True)
                break
        name=spec['name']
        if name in progress['results']:continue
        scene=Path(spec['scene_dir'])
        if hashlib.sha256((scene/'eval_config.yaml').read_bytes()).hexdigest()!=spec['config_sha256']:
            raise ValueError('Scene config changed')
        audit=a.output/'rollouts'/name
        env=os.environ.copy();env.update(DOCKER_HOST=f'unix:///run/user/{os.getuid()}/docker.sock',
            AIC_TRANSPORT_EVAL_GPU=a.gpu,AIC_TRANSPORT_EVAL_POLICY_GPU=a.gpu,
            AIC_TRANSPORT_EVAL_SCENE_DIR=str(scene.resolve()),
            AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA='.06',
            AIC_TRANSPORT_EVAL_POLICY='aic_example_policies.ros.CheatCode',
            AIC_TRANSPORT_EVAL_TEACHER_PREFIX='true',
            AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC='70',
            AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC='200')
        command=['bash','scripts/run_clean_transport_eval.sh',name,str(BASE),str(audit)]
        with (a.output/(name+'.launch.log')).open('w') as log:
            try:run=subprocess.run(command,cwd=ROOT,env=env,stdout=log,
                                   stderr=subprocess.STDOUT,timeout=350)
            except subprocess.TimeoutExpired:run=None
        result=dict(returncode=run.returncode if run else None,official_score=score(audit),
                    audit=str(audit),command=command,source_trial=spec['source_trial'])
        progress['results'][name]=result
        progress_path.write_text(json.dumps(progress,indent=2)+'\n')
        print(json.dumps(dict(number=number,name=name,score=result['official_score'],
                              returncode=result['returncode'])),flush=True)
        if run is None or run.returncode or not result['official_score']['available']:
            print(json.dumps(dict(stopped='run_error',name=name)),flush=True)
            break
    done=len(progress['results']);full=sum(x['official_score'].get('tier3',0)>=75
                                      for x in progress['results'].values())
    print(json.dumps(dict(completed=done,full=full,planned=len(planned['scenes']))),flush=True)


if __name__=='__main__':main()
