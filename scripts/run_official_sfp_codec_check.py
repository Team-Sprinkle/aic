#!/usr/bin/env python3
"""Bounded observation-only SFP codec check on released qualification-family starts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/nas/chmin/aic_archives/2026-10-11_official_anchor_route_coverage_v1')
BASE = ROOT/'outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt'
TEMPORAL = Path('/var/tmp/chmin_aic_20260918_act/sfp_history_context_live_20261007/corrected.pt')
IMAGENET = Path('/var/tmp/aic_sc_fix_20260924/resnet18-f37072fd.pth')
MODEL = ROOT/'outputs/experiments/2026-10-12_route_consistent_diffusion_transport_v1/best.pt'
NAMES = ('sfp_trial1_validation_04','sfp_trial2_validation_12','sfp_trial2_validation_13')

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--max-new',type=int,default=3)
    a=p.parse_args()
    if not 0<=a.gpu<=7:p.error('GPU must be 0..7')
    source=json.loads((SOURCE/'plan.json').read_text())
    selected=[next(x for x in source['scenes'] if x['name']==name) for name in NAMES]
    assert all(x['split']=='validation' for x in selected)
    plan=dict(schema='aic_official_sfp_codec_check/v1',source_plan=str(SOURCE/'plan.json'),
              released_config_sha256=source['official_sha256'],scenes=selected,
              checkpoint=str(MODEL),checkpoint_sha256=digest(MODEL),
              sfp_native_jpeg_quality=86,actor_privileged_inputs=False,final_scenes_opened=False)
    a.output.mkdir(parents=True,exist_ok=True)
    path=a.output/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Frozen plan changed')
    if not path.exists():path.write_text(json.dumps(plan,indent=2)+'\n')
    progress_path=a.output/'progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
    for x in selected:
        name=x['name']
        if name in progress:continue
        if a.max_new<=0:break
        scene=Path(x['scene_dir'])
        if digest(scene/'eval_config.yaml')!=x['config_sha256']:raise ValueError(f'Scene changed: {name}')
        audit=a.output/'rollouts'/name
        if audit.exists():raise FileExistsError(f'Inspect incomplete {audit}')
        env=os.environ.copy()
        env.update(DOCKER_HOST=f'unix:///run/user/{os.getuid()}/docker.sock',
                   AIC_TRANSPORT_EVAL_GPU=str(a.gpu),AIC_TRANSPORT_EVAL_POLICY_GPU=str(a.gpu),
                   AIC_TRANSPORT_EVAL_SCENE_DIR=str(scene),
                   AIC_TRANSPORT_EVAL_POLICY='aic_example_policies.ros.RunTemporalTransport',
                   AIC_TEMPORAL_TRANSPORT_CHECKPOINT=str(TEMPORAL),
                   AIC_TEMPORAL_TRANSPORT_IMAGENET=str(IMAGENET),
                   AIC_ROUTE_DIFFUSION_CHECKPOINT=str(MODEL),
                   AIC_TRANSPORT_SFP_NATIVE_JPEG_QUALITY='86',
                   AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA='.06',
                   AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC='40',
                   AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC='400',AIC_SFP_DIAGNOSTIC_STOP='0')
        command=['bash','scripts/run_clean_transport_eval.sh',name,str(BASE),str(audit)]
        with (a.output/(name+'.launch.log')).open('w') as log:
            run=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=650)
        progress[name]=dict(returncode=run.returncode,audit=str(audit),command=command)
        progress_path.write_text(json.dumps(progress,indent=2)+'\n')
        print(json.dumps(dict(name=name,returncode=run.returncode)),flush=True)
        a.max_new-=1
        if run.returncode:raise RuntimeError(f'Rollout failed: {name}')

if __name__=='__main__':main()
