#!/usr/bin/env python3
"""Run the remaining frozen nonfinal SFP side-view scenes sequentially."""
import json
import os
from pathlib import Path
import subprocess


ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1')
OUT=Path('/home/nas/chmin/aic_archives/2026-10-04_sfp_side_view_v1')
BASE=ROOT/'outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt'
CASES=('card2_port0','card0_port1','card4_port1')


def main():
    plan={x['name']:x for x in json.loads((SOURCE/'plan.json').read_text())['scenes']}
    manifest=dict(schema='sfp_side_view_sequential_plan/v1',source_plan=str(SOURCE/'plan.json'),
                  case_order=list(CASES),source_reset_hashes={x:plan[x]['reset_sha256'] for x in CASES},
                  no_training=True,final_scenes_opened=False,gpu_count=1)
    path=OUT/'remaining_plan_v3.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise RuntimeError('Plan changed')
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    env={**os.environ,'DOCKER_HOST':f'unix:///run/user/{os.getuid()}/docker.sock',
         'AIC_TRANSPORT_EVAL_GPU':'1','AIC_TRANSPORT_EVAL_POLICY_GPU':'1',
         'AIC_TRANSPORT_EVAL_POLICY':'aic_example_policies.ros.CheatCodeSideViewDiagnostic',
         'AIC_TRANSPORT_EVAL_TEACHER_PREFIX':'true',
         'AIC_TRANSPORT_EVAL_ORACLE_CONTACTS':'true',
         'AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA':'.06',
         'AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC':'70',
         'AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC':'200'}
    progress_path=OUT/'remaining_progress_v3.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'results':{}}
    for name in CASES:
        output=OUT/f'{name}_v3'
        if name in progress['results']:
            if not (output/'side_view_events.json').exists():raise RuntimeError(f'Missing completed event record {name}')
            continue
        if output.exists():raise FileExistsError(output)
        key_row=next((line for line in Path('/proc/key-users').read_text().splitlines()
                      if line.lstrip().startswith(f'{os.getuid()}:')),None)
        if key_row:
            count,limit=key_row.split()[3].split('/')
            if int(count)>=int(limit):raise RuntimeError(f'Rootless key quota {count}/{limit}')
        env['AIC_TRANSPORT_EVAL_SCENE_DIR']=plan[name]['scene']
        command=['bash','scripts/run_clean_transport_eval.sh',f'side_view_{name}_v3',str(BASE),str(output)]
        with (OUT/f'{name}_v3.launch.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=360,check=True)
        events_path=output/'side_view_events.json'
        events=json.loads(events_path.read_text()) if events_path.exists() else None
        stages={x['stage'] for x in events['events']} if events else set()
        result=dict(scene=plan[name],command=command,events=str(events_path),
                    baseline_view='baseline_view' in stages,
                    positive_view='positive_view' in stages,
                    negative_view='negative_view' in stages,
                    unsafe_abort='unsafe_abort' in stages)
        progress['results'][name]=result
        progress_path.write_text(json.dumps(progress,indent=2)+'\n')
        print(json.dumps(dict(name=name,**{k:result[k] for k in
                                     ('baseline_view','positive_view','negative_view','unsafe_abort')})),flush=True)
        if not events:
            raise RuntimeError(f'No side-view event record for {name}; inspect audit before next run')
        # An unsafe contact is an intended per-scene stop condition. Preserve
        # that scene and continue to the other predeclared configurations.


if __name__=='__main__':main()
