#!/usr/bin/env python3
"""Bounded matched teacher/control/candidate transport on fresh nonfinal starts."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from run_sfp_cheatcode_target_pilot import BASE, ROOT, score

CAL=ROOT/'configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json'
RESNET=Path('/var/tmp/aic_sc_fix_20260924/resnet18-f37072fd.pth')


def run_case(folder,name,scene,policy,checkpoint,gpu):
    out=folder/'rollouts'/name
    if out.exists():raise FileExistsError(out)
    env=os.environ.copy();env.update(
        DOCKER_HOST=f'unix:///run/user/{os.getuid()}/docker.sock',
        AIC_TRANSPORT_EVAL_GPU=str(gpu),AIC_TRANSPORT_EVAL_POLICY_GPU=str(gpu),
        AIC_TRANSPORT_EVAL_SCENE_DIR=str(scene),
        AIC_TRANSPORT_EVAL_POLICY=policy,
        AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA='.06',
        AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC='70',
        AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC='200',
        AIC_SFP_DIAGNOSTIC_STOP='0')
    if policy.endswith('CheatCode'):
        env['AIC_TRANSPORT_EVAL_TEACHER_PREFIX']='true'
    else:
        env['AIC_TEMPORAL_TRANSPORT_CHECKPOINT']=str(checkpoint.resolve())
        env['AIC_TEMPORAL_TRANSPORT_IMAGENET']=str(RESNET)
    command=['bash','scripts/run_clean_transport_eval.sh',name,str(BASE),str(out)]
    launch_log=folder/(name+'.launch.log')
    timed_out=False
    with launch_log.open('w') as log:
        try:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,
                                  stderr=subprocess.STDOUT,timeout=350)
            rc=result.returncode
        except subprocess.TimeoutExpired:
            rc=None;timed_out=True
    if timed_out:
        # The host launcher has exited; only its timed-out container remains.
        docker_env=os.environ.copy();docker_env['DOCKER_HOST']=env['DOCKER_HOST']
        container='aic_clean_transport_eval_'+out.name
        subprocess.run(['docker','stop',container],env=docker_env,stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,timeout=30)
        subprocess.run(['docker','rm',container],env=docker_env,stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,timeout=30)
    return dict(output=str(out),returncode=rc,timed_out=timed_out,
                official_score=score(out),command=command,launch_log=str(launch_log),
                scene=str(scene),checkpoint=str(checkpoint) if checkpoint else None,
                policy=policy)


def extract_teacher(audit):
    output=audit/'scored_stream.json'
    if output.exists():return
    env=os.environ.copy()
    env['PYTHONPATH']='/var/tmp/chmin_aic_mcap_reader_py312'
    env['LD_LIBRARY_PATH']=str(ROOT/'.pixi/envs/default/lib')+':'+env.get('LD_LIBRARY_PATH','')
    subprocess.run([str(ROOT/'.pixi/envs/default/bin/python'),
                    str(ROOT/'scripts/extract_moveit_training_stream.py'),
                    str(audit),str(output)],cwd=ROOT,env=env,check=True,
                   stdout=subprocess.DEVNULL)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--gpu',default='1')
    a=p.parse_args()
    plan=json.loads((a.plan/'plan.json').read_text())
    if plan['schema']!='aic_sfp_paired_actor_live_starts/v1' or plan['final_scenes_opened']:
        p.error('Expected sealed nonfinal plan')
    progress_path=a.plan/'progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'teachers':{},'actors':{}}
    selected=[]
    for count in (2,3,5):
        candidates=[x for x in plan['candidate_scenes'] if x['card_count']==count]
        teacher=None
        for spec in candidates:
            name=spec['name']+'_teacher'
            if name not in progress['teachers']:
                record=run_case(a.plan,name,Path(spec['scene_dir']),
                                'aic_example_policies.ros.CheatCode',None,a.gpu)
                progress['teachers'][name]=record
                progress_path.write_text(json.dumps(progress,indent=2)+'\n')
                print(json.dumps(dict(name=name,score=record['official_score'])),flush=True)
            record=progress['teachers'][name]
            if record['official_score'].get('tier3',0)>=75:
                teacher=record;selected.append(spec);break
        if teacher is None:
            print(json.dumps(dict(card_count=count,no_successful_teacher=True)),flush=True)
            continue
        extract_teacher(Path(teacher['output']))
        for variant,checkpoint in (('baseline',a.baseline),('candidate',a.candidate)):
            name=spec['name']+'_'+variant
            if name not in progress['actors']:
                record=run_case(a.plan,name,Path(spec['scene_dir']),
                                'aic_example_policies.ros.RunTemporalTransport',checkpoint,a.gpu)
                record.update(variant=variant,teacher=teacher['output'])
                progress['actors'][name]=record
                progress_path.write_text(json.dumps(progress,indent=2)+'\n')
                print(json.dumps(dict(name=name,returncode=record['returncode'],
                                      score=record['official_score'])),flush=True)
    if len(selected)!=3:
        raise RuntimeError(f'Only {len(selected)}/3 successful fresh teacher starts')
    command=[str(ROOT/'.pixi/envs/default/bin/python'),
             str(ROOT/'scripts/score_sfp_autonomous_transport.py')]
    for spec in selected:
        for variant in ('baseline','candidate'):
            actor=progress['actors'][spec['name']+'_'+variant]
            if actor['returncode'] or not (Path(actor['output'])/'results/scoring.yaml').exists():
                raise RuntimeError(f'Failed actor rollout: {actor}')
            command.extend(('--run',actor['output'],'--teacher',actor['teacher']))
    command.extend(('--calibration',str(CAL),'--output',str(a.plan/'posthoc_score.json')))
    subprocess.run(command,cwd=ROOT,check=True)
    scored=json.loads((a.plan/'posthoc_score.json').read_text())['runs']
    results={}
    for variant in ('baseline','candidate'):
        items=[x for x in scored if Path(x['run']).name.endswith('_'+variant)]
        results[variant]=dict(starts=len(items),safe_buffered_arrivals=sum(x['buffered_visit'] for x in items),
                              p95_inference_ms=max(x['inference_p50_p95_p99_ms'][1] for x in items),
                              minimum_official_contact_score=min(x['official_contact_score'] for x in items))
    summary=dict(schema='aic_sfp_paired_actor_fresh_live_gate/v1',
                 selected_scene_groups=[x['scene_digest'] for x in selected],
                 results=results,gate_passed=bool(results['candidate']['safe_buffered_arrivals']>=2 and
                     results['candidate']['safe_buffered_arrivals']>=results['baseline']['safe_buffered_arrivals'] and
                     results['candidate']['p95_inference_ms']<300 and
                     results['candidate']['minimum_official_contact_score']>=
                     results['baseline']['minimum_official_contact_score']),
                 no_guide_or_privileged_actor_input=True,automatic_handoff_enabled=False,
                 final_scenes_opened=False)
    (a.plan/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
