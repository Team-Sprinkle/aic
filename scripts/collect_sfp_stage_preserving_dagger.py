#!/usr/bin/env python3
"""Bounded five-card SFP on-policy prefix / successful teacher continuation pilot.

Teacher control and true geometry start only after the saved autonomous actor
prefix. Only official full insertion with a retained takeover timestamp is
eligible for supervised corrective training. No final scene is opened.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

import yaml


IMAGE='ghcr.io/intrinsic-dev/aic/aic_eval@sha256:9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=3)
    p.add_argument('--max-attempts',type=int,default=14)
    p.add_argument('--max-repeats-per-case',type=int,default=2)
    p.add_argument('--decision-limit',type=int,default=20)
    p.add_argument('--cases',nargs='+',
                   help='Optional exact non-final case names for a bounded phase probe')
    a=p.parse_args()
    repo=Path.cwd().resolve();archive=a.output.resolve()
    if archive.exists():p.error('Preserve prior collection')
    archive.mkdir(parents=True)
    all_names=[f'card{card}_port{port}_cards5_train'
               for card in range(5) for port in range(2)]
    names=a.cases if a.cases else all_names
    if len(names)!=len(set(names)) or any(name not in all_names for name in names):
        p.error('--cases must contain distinct allowed five-card training cases')
    source=repo/'outputs/experiments/2026-10-02_two_family_perception/sfp_native_scene_selection_v1'
    for name in names:
        if not (source/name/'results/scoring.yaml').is_file():
            raise FileNotFoundError(source/name)
    env=os.environ.copy()
    env.update(DOCKER_HOST=f'unix:///run/user/{os.getuid()}/docker.sock',
        AIC_TRANSPORT_EVAL_GPU=str(a.gpu),AIC_TRANSPORT_EVAL_POLICY_GPU=str(a.gpu),
        AIC_SFP_POST_HANDOFF_STAGE_PRESERVING_TEACHER='1',
        AIC_SFP_STAGE_PRESERVE_AFTER_LOCAL='1',
        AIC_SFP_LOCAL_DECISION_LIMIT=str(a.decision_limit),
        AIC_LOCAL_SFP_PAIR_ORIENTATION='1',AIC_LOCAL_SFP_LOCK_SAMPLES='1',
        AIC_SFP_PHYSICAL_STANDOFF_M='.085',AIC_SFP_PHYSICAL_LATERAL_MAX_M='.002')
    attempts=[];completed=[]
    for case in names:
        if len(attempts)>=a.max_attempts:break
        for repeat in range(1,a.max_repeats_per_case+1):
            if len(attempts)>=a.max_attempts:break
            audit=archive/f'{case}_r{repeat}'
            env['AIC_SFP_PHYSICAL_SCENE']=case
            env['AIC_SFP_PHYSICAL_HANDOFF_AUDIT']=str(audit)
            with (archive/f'{case}_r{repeat}.launch.log').open('w') as log:
                code=subprocess.run(['bash','scripts/run_sfp_actor39_physical_tip_handoff.sh'],
                    cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            scoring=audit/'results/scoring.yaml'
            tier=None
            if scoring.is_file():
                result=yaml.safe_load(scoring.read_text())
                trial=next(k for k in result if k.startswith('trial_'))
                tier=float(result[trial]['tier_3']['score'])
            events=audit/'teacher_takeover.json'
            handoff=audit/'physical_tip_handoff.json'
            prefix=0
            trace=audit/'runtime_trace.jsonl'
            if trace.is_file():
                prefix=sum(json.loads(line).get('event')=='command' for line in trace.open())
            record={'case':case,'repeat':repeat,'audit':str(audit),'exit_code':code,
                    'tier3':tier,'handoff':handoff.is_file(),
                    'takeover_event':events.is_file(),'actor_commands':prefix,
                    'eligible':code==0 and tier==75 and handoff.is_file() and
                               events.is_file() and prefix>0}
            attempts.append(record)
            if record['eligible']:
                stream=audit/'scored_stream.json'
                docker=['docker','run','--rm','--net=host','--entrypoint','/bin/bash',
                    '-v',f'{repo}:{repo}:ro','-v',f'{audit}:/audit','-w',str(repo),IMAGE,
                    '-lc','source /ws_aic/install/setup.bash; python3 scripts/extract_moveit_training_stream.py /audit /audit/scored_stream.json']
                with (archive/f'{case}_r{repeat}.extract.log').open('w') as log:
                    extract=subprocess.run(docker,cwd=repo,env=env,stdout=log,
                                           stderr=subprocess.STDOUT).returncode
                if extract:raise RuntimeError(f'Scored-stream extraction failed: {audit}')
                text=stream.read_text().replace('/audit/frames/',str(audit/'frames')+'/')
                stream.write_text(text)
                completed.append(record)
            print(json.dumps(record),flush=True)
            summary={'schema':'aic_sfp_stage_preserving_dagger_collection/v1',
                'attempts':attempts,'full_successes':completed,
                'attempt_cap':a.max_attempts,'decision_limit':a.decision_limit,
                'selected_cases':names,
                'rootless':True,'gpu':a.gpu,'reserved_final_opened':False}
            (archive/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            if record['eligible']:break
        # If five distinct identities fail to produce three admitted successes,
        # stop instead of gathering routine failures or retraining blindly.
        tested=len({row['case'] for row in attempts})
        if tested>=5 and len(completed)<3:
            summary['stop_reason']='five_case_success_gate_failed'
            (archive/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            print(json.dumps({'stop_reason':summary['stop_reason']}),flush=True)
            break
    else:
        summary['stop_reason']='ten_identity_cap'
        (archive/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'attempts':len(attempts),'full_successes':len(completed),
                      'cases':len({row['case'] for row in completed})}),flush=True)


if __name__=='__main__':main()
