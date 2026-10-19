#!/usr/bin/env python3
"""Frozen deterministic temporal-transformer arm on the public-anchor development starts."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from run_official_anchor_route_match import BASE,TEMPORAL,IMAGENET,SCENES,digest,ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--general',type=Path,required=True)
    p.add_argument('--exact',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=3)
    p.add_argument('--policy-gpu',type=int,default=4)
    a=p.parse_args()
    if a.gpu==a.policy_gpu:p.error('Separate simulator and actor GPUs required')
    plans={key:json.loads((folder/'plan.json').read_text()) for key,folder in
           (('general',a.general),('exact',a.exact))}
    selected=[]
    for key,name in SCENES:
        scene=next(s for s in plans[key]['scenes'] if s['name']==name)
        if scene['split']!='validation':raise ValueError('Development split required')
        selected.append(scene)
    frozen=dict(schema='aic_official_anchor_transformer_match/v1',scenes=selected,
                transformer_checkpoint=str(TEMPORAL),transformer_sha256=digest(TEMPORAL),
                base_checkpoint=str(BASE),base_sha256=digest(BASE),
                sim_gpu=a.gpu,policy_gpu=a.policy_gpu,simulation_budget_s=40,
                actor_privileged_inputs=False,teacher_or_guide=False,
                automatic_handoff=False,final_scenes_opened=False)
    a.output.mkdir(parents=True,exist_ok=True)
    plan_path=a.output/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=frozen:
        raise ValueError('Frozen transformer comparison changed')
    if not plan_path.exists():plan_path.write_text(json.dumps(frozen,indent=2)+'\n')
    progress_path=a.output/'progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
    for scene in selected:
        name=scene['name']
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
            AIC_ROUTE_DIFFUSION_CHECKPOINT='',
            AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA='.06',
            AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC='40',
            AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC='240',
            AIC_SFP_DIAGNOSTIC_STOP='0')
        cmd=['bash','scripts/run_clean_transport_eval.sh','transformer_'+name,str(BASE),str(audit)]
        with (a.output/(name+'.launch.log')).open('w') as log:
            run=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,
                               stderr=subprocess.STDOUT,timeout=600)
        progress[name]=dict(returncode=run.returncode,audit=str(audit),command=cmd)
        progress_path.write_text(json.dumps(progress,indent=2)+'\n')
        print(json.dumps(dict(scene=name,returncode=run.returncode)),flush=True)
        if run.returncode:raise RuntimeError(f'Transformer rollout failed: {name}')


if __name__=='__main__':main()
