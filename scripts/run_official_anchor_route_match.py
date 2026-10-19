#!/usr/bin/env python3
"""Frozen observation-only paired transport evaluation on public development starts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt'
TEMPORAL=Path('/var/tmp/chmin_aic_20260918_act/sfp_history_context_live_20261007/corrected.pt')
IMAGENET=Path('/var/tmp/aic_sc_fix_20260924/resnet18-f37072fd.pth')
SCENES=(('general','sfp_trial1_validation_04'),
        ('general','sfp_trial2_validation_12'),
        ('general','sfp_trial2_validation_13'),
        ('general','sc_trial3_validation_17'),
        ('general','sc_trial3_validation_18'),
        ('exact','sc_trial3_exact_validation_06'))


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--general',type=Path,required=True)
    p.add_argument('--exact',type=Path,required=True)
    p.add_argument('--control',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=3)
    p.add_argument('--policy-gpu',type=int,default=4)
    p.add_argument('--max-pairs',type=int,default=6)
    a=p.parse_args()
    if a.gpu==a.policy_gpu or not all(0<=x<=7 for x in (a.gpu,a.policy_gpu)):
        p.error('Use distinct valid physical GPUs for simulator and actor')
    a.output.mkdir(parents=True,exist_ok=True)
    plans={key:json.loads((folder/'plan.json').read_text()) for key,folder in
           (('general',a.general),('exact',a.exact))}
    selected=[]
    for key,name in SCENES:
        scene=next(s for s in plans[key]['scenes'] if s['name']==name)
        if scene['split']!='validation':raise ValueError('Development scene expected')
        selected.append(scene)
    checkpoints={'control':a.control.resolve(),'candidate':a.candidate.resolve()}
    frozen=dict(schema='aic_official_anchor_route_match/v1',
                scenes=selected,checkpoints={k:dict(path=str(v),sha256=digest(v)) for k,v in checkpoints.items()},
                sim_gpu=a.gpu,policy_gpu=a.policy_gpu,simulation_budget_s=40,
                actor_privileged_inputs=False,teacher_or_guide=False,automatic_handoff=False,
                final_scenes_opened=False)
    plan_path=a.output/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=frozen:
        raise ValueError('Frozen comparison changed')
    if not plan_path.exists():plan_path.write_text(json.dumps(frozen,indent=2)+'\n')
    progress_path=a.output/'progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
    for scene in selected[:a.max_pairs]:
        for arm,cp in checkpoints.items():
            name=arm+'_'+scene['name']
            if name in progress:continue
            folder=Path(scene['scene_dir'])
            if digest(folder/'eval_config.yaml')!=scene['config_sha256']:
                raise ValueError(f'Scene changed: {name}')
            audit=a.output/'rollouts'/name
            if audit.exists():raise FileExistsError(f'Inspect incomplete {audit}')
            env=os.environ.copy();env.update(
                DOCKER_HOST=f'unix:///run/user/{os.getuid()}/docker.sock',
                AIC_TRANSPORT_EVAL_GPU=str(a.gpu),AIC_TRANSPORT_EVAL_POLICY_GPU=str(a.policy_gpu),
                AIC_TRANSPORT_EVAL_SCENE_DIR=str(folder),
                AIC_TRANSPORT_EVAL_POLICY='aic_example_policies.ros.RunTemporalTransport',
                AIC_TEMPORAL_TRANSPORT_CHECKPOINT=str(TEMPORAL),
                AIC_TEMPORAL_TRANSPORT_IMAGENET=str(IMAGENET),
                AIC_ROUTE_DIFFUSION_CHECKPOINT=str(cp),
                AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA='.06',
                AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC='40',
                AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC='240',AIC_SFP_DIAGNOSTIC_STOP='0')
            command=['bash','scripts/run_clean_transport_eval.sh',name,str(BASE),str(audit)]
            with (a.output/(name+'.launch.log')).open('w') as log:
                run=subprocess.run(command,cwd=ROOT,env=env,stdout=log,
                                   stderr=subprocess.STDOUT,timeout=600)
            progress[name]=dict(scene=scene['name'],arm=arm,returncode=run.returncode,
                                audit=str(audit),command=command)
            progress_path.write_text(json.dumps(progress,indent=2)+'\n')
            print(json.dumps(dict(name=name,returncode=run.returncode)),flush=True)
            if run.returncode:raise RuntimeError(f'Rollout failed: {name}')


if __name__=='__main__':main()
