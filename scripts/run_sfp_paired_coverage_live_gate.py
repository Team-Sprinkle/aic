#!/usr/bin/env python3
"""Matched nonfinal SFP transport rollouts for the frozen A and paired-data actor."""
import argparse
import json
import os
from pathlib import Path
import subprocess


ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt'
RESNET=Path('/var/tmp/aic_sc_fix_20260924/resnet18-f37072fd.pth')
CAL=ROOT/'configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',default='1')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    plan=json.loads((a.collection/'plan.json').read_text())
    if plan.get('source_split')!='development' or plan['final_scenes_opened']:
        p.error('Expected nonfinal development plan')
    progress=json.loads((a.collection/'progress.json').read_text())['results']
    selected=[]
    for count,amount in ((1,1),(2,2)):
        pool=[r for r in plan['scenes'] if r['card_count']==count and
              progress[r['name']]['official_score'].get('tier3',0)>=75 and
              (Path(progress[r['name']]['audit'])/'scored_stream.json').exists()]
        if len(pool)<amount:p.error(f'Insufficient scored teachers for {count} cards')
        selected.extend(pool[:amount])
    run_path=a.output/'progress.json'
    runs=json.loads(run_path.read_text()) if run_path.exists() else {'runs':{}}
    for spec in selected:
        for variant,checkpoint in (('baseline',a.baseline),('candidate',a.candidate)):
            name=f'{spec["name"]}_{variant}'
            if name in runs['runs']:continue
            out=a.output/name
            if out.exists():raise ValueError(f'Untracked rollout exists: {out}')
            env=os.environ.copy();env.update(
                DOCKER_HOST=f'unix:///run/user/{os.getuid()}/docker.sock',
                AIC_TRANSPORT_EVAL_GPU=a.gpu,AIC_TRANSPORT_EVAL_POLICY_GPU=a.gpu,
                AIC_TRANSPORT_EVAL_SCENE_DIR=spec['scene_dir'],
                AIC_TRANSPORT_EVAL_POLICY='aic_example_policies.ros.RunTemporalTransport',
                AIC_TEMPORAL_TRANSPORT_CHECKPOINT=str(checkpoint.resolve()),
                AIC_TEMPORAL_TRANSPORT_IMAGENET=str(RESNET),
                AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA='.06',
                AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC='70',
                AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC='200',
                AIC_SFP_DIAGNOSTIC_STOP='0')
            command=['bash','scripts/run_clean_transport_eval.sh',name,str(BASE),str(out)]
            with (a.output/(name+'.launch.log')).open('w') as log:
                result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,
                                      stderr=subprocess.STDOUT,timeout=350)
            record=dict(variant=variant,scene=spec['scene_dir'],
                        teacher=progress[spec['name']]['audit'],
                        checkpoint=str(checkpoint.resolve()),output=str(out),
                        returncode=result.returncode,command=command)
            runs['runs'][name]=record
            run_path.write_text(json.dumps(runs,indent=2)+'\n')
            print(json.dumps(dict(name=name,returncode=result.returncode)),flush=True)
            if result.returncode or not (out/'results/scoring.yaml').exists():
                raise RuntimeError(f'Failed actor rollout: {name}')
    cmd=[str(ROOT/'.pixi/envs/default/bin/python'),
         str(ROOT/'scripts/score_sfp_autonomous_transport.py')]
    for spec in selected:
        for variant in ('baseline','candidate'):
            row=runs['runs'][f'{spec["name"]}_{variant}']
            cmd.extend(('--run',row['output'],'--teacher',row['teacher']))
    cmd.extend(('--calibration',str(CAL),'--output',str(a.output/'posthoc_score.json')))
    subprocess.run(cmd,cwd=ROOT,check=True)
    summary=json.loads((a.output/'posthoc_score.json').read_text())
    counts={variant:sum(r['buffered_visit'] for r in summary['runs']
                        if Path(r['run']).name.endswith('_'+variant))
            for variant in ('baseline','candidate')}
    (a.output/'summary.json').write_text(json.dumps(dict(
        starts=len(selected),buffered_visits=counts,source=summary,
        automatic_handoff_enabled=False,final_scenes_opened=False),indent=2)+'\n')
    print(json.dumps(dict(starts=len(selected),buffered_visits=counts)),flush=True)


if __name__=='__main__':main()
