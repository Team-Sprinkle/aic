#!/usr/bin/env python3
"""Observation-only transport rollouts on held-out teacher-generator scenes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import yaml

ROOT=Path(__file__).resolve().parents[1]
SFP=Path('/home/nas/chmin/aic_archives/2026-10-12_sfp_start_variants_v1')
SC=ROOT/'outputs/trajectory_datasets/sc_llb_cleaned_v3/manifest.json'
BASE=ROOT/'outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt'
TEMPORAL=Path('/var/tmp/chmin_aic_20260918_act/sfp_history_context_live_20261007/corrected.pt')
IMAGENET=Path('/var/tmp/aic_sc_fix_20260924/resnet18-f37072fd.pth')
MODEL=ROOT/'outputs/experiments/2026-10-12_route_consistent_diffusion_transport_v1/best.pt'
SFP_NAMES=('sfp_trial1_validation_04','sfp_trial2_validation_12','sfp_trial2_validation_14')
SC_NAMES=('llb_0005','llb_0074','llb_0183')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(model=MODEL, sfp_names=SFP_NAMES, sc_names=SC_NAMES,
         sc_training_diagnostic=False):
    sfp=json.loads((SFP/'plan.json').read_text())
    sc=json.loads(SC.read_text())
    items=[]
    for name in sfp_names:
        x=next(x for x in sfp['scenes'] if x['name']==name)
        assert x['split']=='validation'
        scene=Path(x['scene_dir']);teacher=SFP/'rollouts'/name
        items.append(dict(name=name,family='sfp',scene=str(scene),teacher=str(teacher),
                          scene_sha256=sha(scene/'eval_config.yaml'),simulation_s=40))
    for name in sc_names:
        x=next(x for x in sc['episodes'] if x['case']==name)
        assert x['split']==('train' if sc_training_diagnostic else 'validation')
        scene=Path(x['source_episode'])
        items.append(dict(name=name,family='sc',scene=str(scene),teacher=str(scene),
                          scene_sha256=sha(scene/'eval_config.yaml'),simulation_s=70,
                          card_count=x['card_count'],selected_port=x['selected_port']))
    for x in items:
        score=yaml.safe_load((Path(x['teacher'])/'results/scoring.yaml').read_text())
        trial=next(v for k,v in score.items() if k.startswith('trial_'))
        assert trial['tier_3']['score']>=75
        assert trial['tier_2']['categories']['contacts']['message']=='No contact detected.'
    return dict(schema='aic_teacher_distribution_route_diagnostic/v1',scenes=items,
                checkpoint=str(model),checkpoint_sha256=sha(model),
                sc_training_diagnostic=sc_training_diagnostic,
                actor_privileged_inputs=False,final_scenes_opened=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--policy-gpu',type=int)
    p.add_argument('--checkpoint',type=Path,default=MODEL)
    p.add_argument('--max-new',type=int,default=6)
    p.add_argument('--only',action='append',
                   help='Run only selected validation scenes in the frozen plan')
    p.add_argument('--sc-name',action='append',
                   help='Override the default SC development scenes; repeat for several names')
    p.add_argument('--sc-only',action='store_true',
                   help='Exclude SFP scenes from this frozen development plan')
    p.add_argument('--sc-training-diagnostic',action='store_true',
                   help='Use SC train scenes for an explicitly non-generalization diagnostic')
    a=p.parse_args()
    if a.sc_training_diagnostic and not a.sc_only:
        p.error('--sc-training-diagnostic requires --sc-only')
    if not 0<=a.gpu<=7:p.error('GPU must be 0..7')
    a.output.mkdir(parents=True,exist_ok=True)
    model=a.checkpoint.resolve()
    if not model.is_file():p.error('Checkpoint does not exist')
    sfp_names=() if a.sc_only else SFP_NAMES
    sc_names=tuple(a.sc_name) if a.sc_name else SC_NAMES
    if len(set(sc_names))!=len(sc_names):p.error('Duplicate SC scene name')
    selected=(*sfp_names,*sc_names)
    if a.only and not set(a.only)<=set(selected):
        p.error('--only scene is not in the selected development plan')
    frozen=plan(model,sfp_names,sc_names,a.sc_training_diagnostic);path=a.output/'plan.json'
    if path.exists() and json.loads(path.read_text())!=frozen:
        raise ValueError('Frozen teacher-distribution plan changed')
    if not path.exists():path.write_text(json.dumps(frozen,indent=2)+'\n')
    progress_path=a.output/'progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
    new=0
    for x in frozen['scenes']:
        if a.only and x['name'] not in a.only:continue
        if x['name'] in progress:continue
        if new>=a.max_new:break
        scene=Path(x['scene']);assert sha(scene/'eval_config.yaml')==x['scene_sha256']
        audit=a.output/'rollouts'/x['name']
        if audit.exists():raise FileExistsError(f'Inspect incomplete {audit}')
        env=os.environ.copy();env.update(
            DOCKER_HOST=f'unix:///run/user/{os.getuid()}/docker.sock',
            AIC_TRANSPORT_EVAL_GPU=str(a.gpu),AIC_TRANSPORT_EVAL_POLICY_GPU=str(a.policy_gpu if a.policy_gpu is not None else a.gpu),
            AIC_TRANSPORT_EVAL_SCENE_DIR=str(scene),
            AIC_TRANSPORT_EVAL_POLICY='aic_example_policies.ros.RunTemporalTransport',
            AIC_TEMPORAL_TRANSPORT_CHECKPOINT=str(TEMPORAL),
            AIC_TEMPORAL_TRANSPORT_IMAGENET=str(IMAGENET),
            AIC_ROUTE_DIFFUSION_CHECKPOINT=str(model),
            AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA='.06',
            AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC=str(x['simulation_s']),
            AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC='400',AIC_SFP_DIAGNOSTIC_STOP='0')
        cmd=['bash','scripts/run_clean_transport_eval.sh',x['name'],str(BASE),str(audit)]
        with (a.output/(x['name']+'.launch.log')).open('w') as log:
            run=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=650)
        progress[x['name']]=dict(returncode=run.returncode,audit=str(audit),command=cmd)
        progress_path.write_text(json.dumps(progress,indent=2)+'\n')
        print(json.dumps(dict(name=x['name'],returncode=run.returncode)),flush=True)
        new+=1
        if run.returncode:raise RuntimeError(f'Rollout failed: {x["name"]}')


if __name__=='__main__':main()
