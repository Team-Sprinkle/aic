#!/usr/bin/env python3
"""Six targeted, unused five-card SFP training scenes for coarse identity coverage.

The first three are a feasibility gate; continue only if >=2 fully insert.
No source agent actions are used. Outputs are immutable/resumable.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess

from cache_sfp_balanced_arrival_features import source_episodes
from run_sfp_cheatcode_target_pilot import BASE,TEMPLATE,card_count,score

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/trajectory_datasets/clean_including_no_insert_trajs/sfp_to_nic/agent/nic_cards_mixed/n400__ec2_02_sfp_hard'
CAL=ROOT/'configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json'
OUTPUT=Path('/home/nas/chmin/aic_archives/2026-10-04_sfp_coarse_targeted_coverage_v1')
IDS=('000305','000492','000464','000402','000453','000514')

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    eps=source_episodes(SOURCE,json.loads(CAL.read_text()))
    by_id={Path(e['trial']).stem.removeprefix('trial_'):e for e in eps}
    existing=set()
    for path in (Path('/home/nas/chmin/aic_archives/2026-10-04_sfp_requested_history_extra5_labels_v1/rows.jsonl'),
                 Path('/home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/labels_all_ten/rows.jsonl')):
        for line in path.open():existing.add(json.loads(line)['scene_group'])
    chosen=[]
    for tid in IDS:
        e=by_id[tid]
        if e['split']!='train' or card_count(e['trial'])!=5 or e['scene'] in existing:
            raise ValueError(f'Invalid source or scene overlap: {tid}')
        chosen.append(e)
    plan={'schema':'sfp_coarse_targeted_coverage/v1','purpose':'five-card middle-port identity coverage',
          'source_action_arrays_used':False,'final_scenes_opened':False,'first_three_full_gate':2,
          'scenes':[dict(name=f'card{e["card"]}_port{e["port"]}_trial_{Path(e["trial"]).stem.removeprefix("trial_")}',
                         card=e['card'],port=e['port'],card_count=5,reset_sha256=e['scene'],
                         source_trial=str(e['trial']),source_split=e['split']) for e in chosen]}
    plan_path=OUTPUT/'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text())!=plan:raise ValueError('Existing plan changed')
    else:plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    progress_path=OUTPUT/'progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'results':{}}
    for index,spec in enumerate(plan['scenes']):
        if index==3:
            pilot=[progress['results'].get(s['name'],{}).get('official_score',{})
                   for s in plan['scenes'][:3]]
            if sum(float(v.get('tier3',0))>=75 for v in pilot)<2:
                print(json.dumps({'pilot_failed':True,'scores':pilot}),flush=True)
                break
        name=spec['name']
        if name in progress['results']:continue
        trial=ROOT/spec['source_trial']
        scene=OUTPUT/'scenes'/name;scene.mkdir(parents=True,exist_ok=True)
        shutil.copy2(trial,scene/'eval_config.yaml')
        for item in ('world_audit.sdf','bridge_audit.yaml'):
            shutil.copy2(TEMPLATE/item,scene/item)
        audit=OUTPUT/f'teacher_{name}'
        env=os.environ.copy();env.update(DOCKER_HOST=f'unix:///run/user/{os.getuid()}/docker.sock',
            AIC_TRANSPORT_EVAL_GPU='1',AIC_TRANSPORT_EVAL_POLICY_GPU='1',
            AIC_TRANSPORT_EVAL_SCENE_DIR=str(scene.resolve()),
            AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA='.06',
            AIC_TRANSPORT_EVAL_POLICY='aic_example_policies.ros.CheatCode',
            AIC_TRANSPORT_EVAL_TEACHER_PREFIX='true',
            AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC='70',
            AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC='200')
        cmd=['bash','scripts/run_clean_transport_eval.sh',name,str(BASE),str(audit)]
        with (OUTPUT/f'{name}.launch.log').open('w') as f:
            result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=350)
        stream=audit/'scored_stream.json'
        if result.returncode==0 and not stream.exists():
            extract_env=env.copy()
            extract_env['PYTHONPATH']='/var/tmp/chmin_aic_mcap_reader_py312'
            extract_env['LD_LIBRARY_PATH']=str(ROOT/'.pixi/envs/default/lib')+':'+env.get('LD_LIBRARY_PATH','')
            with (OUTPUT/f'{name}.extract.log').open('w') as f:
                ex=subprocess.run([str(ROOT/'.pixi/envs/default/bin/python'),
                    'scripts/extract_moveit_training_stream.py',str(audit),str(stream)],
                    cwd=ROOT,env=extract_env,stdout=f,stderr=subprocess.STDOUT,timeout=200)
            if ex.returncode:raise RuntimeError(f'Extraction failed: {name}')
        out={'source_trial':spec['source_trial'],'reset_sha256':spec['reset_sha256'],
             'audit':str(audit),'returncode':result.returncode,'official_score':score(audit),
             'stream':str(stream) if stream.exists() else None}
        progress['results'][name]=out
        progress_path.write_text(json.dumps(progress,indent=2)+'\n')
        print(json.dumps({'name':name,'returncode':result.returncode,'score':out['official_score']}),flush=True)
        if result.returncode or not stream.exists():break

if __name__=='__main__':main()
