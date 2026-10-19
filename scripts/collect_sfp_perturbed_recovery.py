#!/usr/bin/env python3
"""Bounded privileged SFP recovery-data collection on selected training resets."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import yaml


IMAGE='ghcr.io/intrinsic-dev/aic/aic_eval@sha256:9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--storage',type=Path,required=True)
    p.add_argument('--case',action='append',required=True)
    p.add_argument('--gpu',type=int,default=3)
    p.add_argument('--allow-development',action='store_true',
                   help='Collect new development resets for evaluation only')
    a=p.parse_args()
    if a.gpu not in range(4):p.error('This experiment is limited to project GPUs 0..3')
    repo=Path.cwd().resolve();selection=a.selection.resolve();storage=a.storage.resolve()
    storage.mkdir(parents=True,exist_ok=True)
    allowed={'train','development'} if a.allow_development else {'train'}
    candidates={r['case']:r for r in json.loads((selection/'collection_manifest.json').read_text())['rows']
                if r['split'] in allowed and 'extra0' in r['case']}
    env=os.environ.copy();env['DOCKER_HOST']='unix:///run/user/1008/docker.sock'
    env.update(AIC_TRANSPORT_EVAL_GPU=str(a.gpu),
        AIC_TRANSPORT_EVAL_POLICY='aic_example_policies.ros.CheatCodePerturbNearPort',
        AIC_TRANSPORT_EVAL_TEACHER_PREFIX='true',
        AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA='.02',
        AIC_NEAR_PERTURB_LATERAL_M='.003',AIC_NEAR_PERTURB_YAW_DEG='4',
        AIC_NEAR_PERTURB_HOLD_S='.75')
    done=[]
    for case in a.case:
        if case not in candidates:raise ValueError(f'Not a selected training reset: {case}')
        r=candidates[case];sign=1 if (r['selected_card']+r['selected_port'])%2==0 else -1
        audit=storage/(case+'_perturb');env['AIC_NEAR_PERTURB_SIGN']=str(sign)
        env['AIC_TRANSPORT_EVAL_SCENE_DIR']=str(selection/case)
        if audit.exists():raise FileExistsError(f'Preserve existing run: {audit}')
        command=['bash','scripts/run_clean_transport_eval.sh',case,
                 str(repo/'outputs/experiments/2026-10-02_two_family_perception/local_actor_sfp_native_29_v4_ensemble/best.pt'),
                 str(audit)]
        with (storage/(case+'.launch.log')).open('w') as log:
            code=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        if code:raise RuntimeError(f'Gazebo launch failed: {case}')
        event=audit/'near_port_perturbation.json'
        if not event.is_file():raise RuntimeError(f'Perturbation not applied: {case}')
        scoring=yaml.safe_load((audit/'results/scoring.yaml').read_text())
        trial=next(k for k in scoring if k.startswith('trial_'))
        tier3=float(scoring[trial]['tier_3']['score'])
        result={'case':case,'split':r['split'],'audit':str(audit),'sign':sign,'tier3':tier3,
                'event':str(event),'scene_group':r['scene_reset_sha256']}
        done.append(result)
        print(json.dumps(result),flush=True)
        (storage/'collection_summary.json').write_text(json.dumps({
            'schema':'aic_sfp_perturbed_recovery_collection/v1','completed':done,
            'gpu':a.gpu,'rootless':True,'reserved_final_opened':False},indent=2)+'\n')
        if tier3!=75:raise RuntimeError(f'Corrective teacher did not fully insert: {case}')
        docker=['docker','run','--rm','--net=host','--entrypoint','/bin/bash',
                '-v',f'{repo}:{repo}:ro','-v',f'{audit}:/audit','-w',str(repo),IMAGE,
                '-lc','source /ws_aic/install/setup.bash; python3 scripts/extract_moveit_training_stream.py /audit /audit/scored_stream.json']
        with (storage/(case+'.extract.log')).open('w') as log:
            code=subprocess.run(docker,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        if code:raise RuntimeError(f'Scored stream extraction failed: {case}')
        path=audit/'scored_stream.json';content=path.read_text()
        if '/audit/frames/' in content:
            temp=audit/'scored_stream.hostpaths.json'
            temp.write_text(content.replace('/audit/frames/',str(audit/'frames')+'/'))
            temp.replace(path)


if __name__=='__main__':main()
