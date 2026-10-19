#!/usr/bin/env python3
"""Sequential rootless CheatCode feasibility pilot across SFP target identities."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import yaml

from cache_sfp_balanced_arrival_features import source_episodes


ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt'
TEMPLATE=ROOT/'outputs/experiments/2026-10-02_two_family_perception/sfp_native_scene_selection_v1/card1_port0_cards5_train'
IMAGE='ghcr.io/intrinsic-dev/aic/aic_eval@sha256:9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923'


def card_count(path):
    cfg=yaml.safe_load(Path(path).read_text());trial=next(iter(cfg['trials'].values()))
    board=trial['scene']['task_board']
    return sum(bool(board[f'nic_rail_{i}']['entity_present']) for i in range(5))


def score(path):
    p=path/'results/scoring.yaml'
    if not p.exists():return dict(available=False)
    x=yaml.safe_load(p.read_text());trial=next((v for k,v in x.items() if k.startswith('trial_')),{})
    return dict(available=True,total=float(x.get('total',0)),
                tier3=float(trial.get('tier_3',{}).get('score',0)),
                message=trial.get('tier_3',{}).get('message'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--skip-target',action='append',default=[],help='card,port already tested')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    cal=json.loads(a.calibration.read_text());episodes=source_episodes(a.source,cal)
    excluded={tuple(map(int,x.split(','))) for x in a.skip_target}
    chosen={}
    for e in episodes:
        key=(e['card'],e['port'])
        if key in excluded or e['split']!='train':continue
        if key not in chosen or (card_count(e['trial']),e['score'])>(card_count(chosen[key]['trial']),chosen[key]['score']):
            chosen[key]=e
    progress_path=a.output/'pilot_progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {
        'schema':'aic_sfp_cheatcode_target_pilot/v1',
        'purpose':'feasibility control, not autonomous policy performance',
        'reserved_final_opened':False,'results':{}}
    for (card,port),e in sorted(chosen.items()):
        count=card_count(e['trial']);name=f'card{card}_port{port}_cards{count}'
        if name in progress['results']:continue
        scene=a.output/'scenes'/name;scene.mkdir(parents=True,exist_ok=True)
        shutil.copy2(e['trial'],scene/'eval_config.yaml')
        for item in ('world_audit.sdf','bridge_audit.yaml'):
            shutil.copy2(TEMPLATE/item,scene/item)
        audit=a.output/('teacher_'+name);log=a.output/(name+'.launch.log')
        env=os.environ.copy();env.update(DOCKER_HOST=f'unix:///run/user/{os.getuid()}/docker.sock',
            AIC_TRANSPORT_EVAL_GPU='1',AIC_TRANSPORT_EVAL_POLICY_GPU='1',
            AIC_TRANSPORT_EVAL_SCENE_DIR=str(scene.resolve()),
            AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA='.06',
            AIC_TRANSPORT_EVAL_POLICY='aic_example_policies.ros.CheatCode',
            AIC_TRANSPORT_EVAL_TEACHER_PREFIX='true',
            AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC='70',
            AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC='200')
        cmd=['bash','scripts/run_clean_transport_eval.sh',name,str(BASE),str(audit)]
        with log.open('w') as f:
            try:r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=320)
            except subprocess.TimeoutExpired:r=None
        result=dict(card=card,port=port,card_count=count,source_trial=e['trial'],
            source_selected_score=e['score'],scene=str(scene),audit=str(audit),
            command=cmd,returncode=r.returncode if r else None,timed_out=r is None,
            official_score=score(audit))
        progress['results'][name]=result
        progress_path.write_text(json.dumps(progress,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    print(json.dumps(dict(completed=len(progress['results']),
        full_insertions=sum(v['official_score'].get('tier3',0)>=75
                            for v in progress['results'].values()))),flush=True)


if __name__=='__main__':main()
