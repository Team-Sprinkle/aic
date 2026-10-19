#!/usr/bin/env python3
"""Collect a bounded, balanced SFP CheatCode command pilot from train scenes.

Selected agent videos supply *scene configs only*. Their invalid Cartesian
action arrays never enter this collection. Every new label comes from a fresh
rootless ordinary CheatCode rollout and its scored ROS bag.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from cache_sfp_balanced_arrival_features import source_episodes
from run_sfp_cheatcode_target_pilot import BASE, TEMPLATE, card_count, score


ROOT = Path(__file__).resolve().parents[1]
PILOT = Path('/home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/cheatcode_target_pilot')
MATCHED = Path('/home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/teacher_card2_port0_cards5_audit420')


def prior_controls():
    rows = {}
    p = PILOT/'pilot_progress.json'
    if p.exists():
        for v in json.loads(p.read_text())['results'].values():
            rows[Path(v['source_trial']).stem] = dict(source_trial=v['source_trial'],
                card=v['card'],port=v['port'],card_count=v['card_count'],
                audit=v['audit'],official_score=v['official_score'],origin='ten-target feasibility pilot')
    if MATCHED.exists():
        rows['trial_000420'] = dict(source_trial='outputs/trajectory_datasets/clean_including_no_insert_trajs/sfp_to_nic/agent/nic_cards_mixed/n400__ec2_02_sfp_hard/trials/trial_000420.yaml',
            card=2,port=0,card_count=5,audit=str(MATCHED),official_score=score(MATCHED),
            origin='matched card2/port0 control')
    return rows


def choose(episodes,prior,max_per_target,source_split='train'):
    by_target=defaultdict(list)
    for e in episodes:
        if e['split']==source_split:by_target[(e['card'],e['port'])].append(e)
    chosen=[]
    for key in sorted(by_target):
        candidates=by_target[key]
        if len(candidates)<max_per_target:
            raise ValueError(f'{key} has only {len(candidates)} train scenes')
        existing=[e for e in candidates if Path(e['trial']).stem in prior]
        if len(existing)>1:raise ValueError(f'{key}: duplicate existing controls')
        chosen.extend(existing)
        seen_counts={card_count(e['trial']) for e in existing}
        remaining=[e for e in candidates if Path(e['trial']).stem not in prior]
        while len([e for e in chosen if (e['card'],e['port'])==key])<max_per_target:
            remaining.sort(key=lambda e:(card_count(e['trial']) in seen_counts,
                hashlib.sha256(str(e['trial']).encode()).hexdigest()))
            e=remaining.pop(0);chosen.append(e);seen_counts.add(card_count(e['trial']))
    return chosen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-per-target',type=int,default=3)
    p.add_argument('--source-split',choices=('train','development'),default='train')
    a=p.parse_args()
    if not 1<=a.max_per_target<=5:p.error('Use a bounded 1–5 scenes per identity')
    a.output.mkdir(parents=True,exist_ok=True)
    prior=prior_controls() if a.source_split=='train' else {}
    episodes=source_episodes(a.source,json.loads(a.calibration.read_text()))
    chosen=choose(episodes,prior,a.max_per_target,a.source_split)
    plan={'schema':'aic_sfp_balanced_teacher_train/v1','requested_per_target':a.max_per_target,
          'source_split':a.source_split,
          'old_agent_action_arrays_used':False,'reserved_final_opened':False,
          'scene_trials':[dict(card=e['card'],port=e['port'],source_trial=str(e['trial']),
              card_count=card_count(e['trial']),prior_control=Path(e['trial']).stem in prior) for e in chosen]}
    (a.output/f'plan_per_target_{a.max_per_target}.json').write_text(json.dumps(plan,indent=2)+'\n')
    progress_path=a.output/'progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {
        'schema':'aic_sfp_balanced_teacher_train_progress/v1','results':prior.copy()}
    for e in chosen:
        trial=Path(e['trial']);tid=trial.stem
        if tid in progress['results']:continue
        name=f'card{e["card"]}_port{e["port"]}_{tid}'
        scene=a.output/'scenes'/name;scene.mkdir(parents=True,exist_ok=True)
        shutil.copy2(trial,scene/'eval_config.yaml')
        for item in ('world_audit.sdf','bridge_audit.yaml'):shutil.copy2(TEMPLATE/item,scene/item)
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
        result=dict(source_trial=str(trial),card=e['card'],port=e['port'],
            card_count=card_count(trial),audit=str(audit),scene=str(scene),
            origin='new ordinary CheatCode bag',command=cmd,
            returncode=r.returncode if r else None,timed_out=r is None,
            official_score=score(audit))
        progress['results'][tid]=result
        progress_path.write_text(json.dumps(progress,indent=2)+'\n')
        print(json.dumps(dict(trial=tid,card=e['card'],port=e['port'],
            card_count=result['card_count'],score=result['official_score'])),flush=True)
    selected=[progress['results'][Path(e['trial']).stem] for e in chosen]
    successes=sum(v['official_score'].get('tier3',0)>=75 for v in selected)
    summary=dict(selected=len(selected),successful=successes,success_rate=successes/len(selected),
        pass_30_gate=(a.source_split=='train' and len(selected)==30 and successes>=27),
        old_agent_action_arrays_used=False,reserved_final_opened=False)
    (a.output/f'summary_per_target_{a.max_per_target}.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
