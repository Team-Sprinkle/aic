#!/usr/bin/env python3
"""Check teacher recoverability after five seconds of an unchanged actor.

This is privileged diagnosis on already-open development scenes, never an
autonomous-policy result or an actor training source.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

import yaml

from run_sfp_cheatcode_target_pilot import BASE, ROOT, score


RESNET = Path('/var/tmp/aic_sc_fix_20260924/resnet18-f37072fd.pth')


def contact_score(folder):
    path=folder/'results/scoring.yaml'
    if not path.exists():return None
    trial=next((v for k,v in yaml.safe_load(path.read_text()).items() if k.startswith('trial_')), {})
    return float(trial.get('tier_2',{}).get('categories',{}).get('contacts',{}).get('score',0))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--live-plan',type=Path)
    source.add_argument('--train-source',type=Path,
                        help='Verified paired-request train-only source plan')
    source.add_argument('--unseen-source',type=Path,
                        help='Disjoint nonfinal development candidate plan')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--actor-prefix-s',type=int,default=5)
    p.add_argument('--card-counts',type=int,nargs='+',
                   help='Fixed subset for rerunning an invalid launch; default is all configured counts')
    p.add_argument('--request-port',type=int,choices=(0,1),default=0,
                   help='Selected requested port for train-only paired-request sources')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=7)
    p.add_argument('--policy-gpu',type=int,
                   help='Optional separate actor GPU; useful when diffusion shares a renderer')
    a=p.parse_args()
    if not 1<=a.actor_prefix_s<=20:p.error('Actor prefix must be 1..20 s')
    if a.output.exists():p.error('Preserve prior diagnostic')
    selected=[]
    if a.live_plan or a.unseen_source:
        scene_plan=a.live_plan or a.unseen_source
        live=json.loads((scene_plan/'plan.json').read_text())
        if live['final_scenes_opened']:p.error('Final scenes are sealed')
        progress=(json.loads((scene_plan/'progress.json').read_text())
                  if a.live_plan else None)
        for count in (a.card_counts or (2,3,5)):
            found=next((x for x in live['candidate_scenes'] if x['card_count']==count and
                        (progress is None or progress['teachers'].get(x['name']+'_teacher',{}).
                         get('official_score',{}).get('tier3',0)>=75)),None)
            if found is None:p.error(f'No eligible development scene with {count} cards')
            selected.append(found)
        source_plan=scene_plan/'plan.json'
    else:
        train=json.loads((a.train_source/'plan.json').read_text())
        if train['final_scenes_opened'] or not (train.get('train_only') or
                                               train.get('source_episode_grouped')):
            p.error('Expected sealed episode-grouped source')
        progress=json.loads((a.train_source/'progress.json').read_text())['results']
        for count in (a.card_counts or (2,3,4,5)):
            found=next((x for x in train['scenes'] if x.get('source_split','train')=='train' and
                        x['card_count']==count and
                        x['alternate_request'][1]==a.request_port and
                        progress.get(x['name'],{}).get('official_score',{}).get('tier3',0)>=75),None)
            if found is None:p.error(f'No successful port-{a.request_port} teacher source with {count} cards')
            selected.append(dict(name=found['name'],card_count=count,
                                 request=found['alternate_request'],
                                 scene_digest=found['source_scene_hash'],
                                 scene_dir=found['scene_dir']))
        source_plan=a.train_source/'plan.json'
    a.output.mkdir(parents=True)
    manifest=dict(schema='aic_sfp_first_departure_takeover_gate/v1',
                  source_plan=str(source_plan),
                  selected=selected,checkpoint=str(a.checkpoint.resolve()),
                  actor_prefix_s=a.actor_prefix_s,teacher_cap_s=60,max_runs=len(selected),
                  diagnostic_privileged_teacher=True,autonomous_success_claim=False,
                  training_source=bool(a.train_source),final_scenes_opened=False,gpu=a.gpu,
                  policy_gpu=a.policy_gpu if a.policy_gpu is not None else a.gpu,
                  route_diffusion_checkpoint=os.environ.get('AIC_ROUTE_DIFFUSION_CHECKPOINT'))
    (a.output/'plan.json').write_text(json.dumps(manifest,indent=2)+'\n')
    results=[]
    for spec in selected:
        name=spec['name']+f'_actor{a.actor_prefix_s}s_teacher_full'
        audit=a.output/'rollouts'/name
        env=os.environ.copy()
        env.update(DOCKER_HOST=f'unix:///run/user/{os.getuid()}/docker.sock',
                   AIC_TRANSPORT_EVAL_GPU=str(a.gpu),
                   AIC_TRANSPORT_EVAL_POLICY_GPU=str(a.policy_gpu if a.policy_gpu is not None else a.gpu),
                   AIC_TRANSPORT_EVAL_SCENE_DIR=spec['scene_dir'],
                   AIC_TRANSPORT_EVAL_POLICY='aic_example_policies.ros.RunTemporalThenCheatCodeDiagnostic',
                   AIC_TRANSPORT_EVAL_TEACHER_PREFIX='true',
                   AIC_TEMPORAL_TRANSPORT_CHECKPOINT=str(a.checkpoint.resolve()),
                   AIC_TEMPORAL_TRANSPORT_IMAGENET=str(RESNET),
                   AIC_SFP_CORRECTIVE_ACTOR_DURATION_S=str(a.actor_prefix_s),
                   AIC_SFP_CORRECTIVE_TEACHER_DURATION_S='60',
                   AIC_SFP_CORRECTIVE_REQUIRE_FULL_INSERT='1',
                   AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA='.06',
                   AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC='90',
                   AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC='300',
                   AIC_SFP_DIAGNOSTIC_STOP='0')
        command=['bash','scripts/run_clean_transport_eval.sh',name,str(BASE),str(audit)]
        with (a.output/(name+'.launch.log')).open('w') as log:
            try:
                run=subprocess.run(command,cwd=ROOT,env=env,stdout=log,
                                   stderr=subprocess.STDOUT,timeout=500)
                rc=run.returncode
            except subprocess.TimeoutExpired:
                rc=None
        event_path=audit/'corrective_teacher_takeover.json'
        event=json.loads(event_path.read_text()) if event_path.exists() else {}
        official=score(audit);contact=contact_score(audit)
        row=dict(name=name,scene_digest=spec['scene_digest'],card_count=spec['card_count'],
                 request=spec['request'],audit=str(audit),command=command,returncode=rc,
                 takeover_sim_time_s=event.get('takeover_sim_time_s'),
                 teacher_outcome=event.get('teacher_outcome'),
                 official_score=official,official_contact_score=contact,
                 gate_passed=bool(rc==0 and official.get('tier3',0)>=75 and
                                  contact is not None and contact>=0 and
                                  event.get('takeover_sim_time_s') is not None))
        results.append(row)
        summary=dict(schema='aic_sfp_first_departure_takeover_gate/v1',
                     runs=results,completed=len(results),
                     safe_full_insertions=sum(x['gate_passed'] for x in results),
                     valid_takeover_logs=sum(x['takeover_sim_time_s'] is not None for x in results),
                     gate_passed=len(results)==len(selected) and
                                 all(x['takeover_sim_time_s'] is not None for x in results) and
                                 sum(x['gate_passed'] for x in results)>=
                                 (3 if a.train_source and len(selected)>=3 else
                                  2 if len(selected)>=2 else 1),
                     autonomous_success_claim=False,training_source=bool(a.train_source),
                     final_scenes_opened=False)
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(row),flush=True)
        if rc is None:raise RuntimeError(f'Timed out: inspect {audit}')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
