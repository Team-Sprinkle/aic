#!/usr/bin/env python3
"""Capped train-group actor-prefix/teacher-recovery pilot; no policy training."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import yaml


ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/home/nas/chmin/aic_archives/2026-10-07_sfp_divergent_pairs_round2_v1')
BASE=ROOT/'outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt'
TEMPORAL=Path('/var/tmp/chmin_aic_20260918_act/sfp_history_context_live_20261007/corrected.pt')
IMAGENET=Path('/var/tmp/aic_sc_fix_20260924/resnet18-f37072fd.pth')


def point_force(path):
    peak=0.
    if not path.exists():return None
    with path.open() as stream:
        for line in stream:
            row=json.loads(line)
            for contact in row['contacts']:
                names=contact['collision1']['name']+' '+contact['collision2']['name']
                if 'cable_0' not in names or 'task_board' not in names:
                    continue
                for wrench in contact['wrenches']:
                    force=wrench['body_1_wrench']['force']
                    peak=max(peak,float(np.linalg.norm([force[k] for k in 'xyz'])))
    return peak


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diffusion',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=10)
    p.add_argument('--delay-tf',action='store_true')
    p.add_argument('--gpu',type=int,default=4)
    p.add_argument('--policy-gpu',type=int,default=3)
    a=p.parse_args()
    if not 1<=a.limit<=10:p.error('At most ten starts')
    a.output.mkdir(parents=True,exist_ok=True)
    source=json.loads((SOURCE/'plan.json').read_text())
    prior=json.loads((SOURCE/'progress.json').read_text())['results']
    selected=[]
    for count,number in ((2,2),(3,2),(4,2),(5,4)):
        candidates=[r for r in source['scenes'] if r['source_split']=='train'
                    and r['card_count']==count and
                    prior.get(r['name'],{}).get('official_score',{}).get('tier3',0)>=75]
        if len(candidates)<number:raise ValueError(f'Insufficient successful teacher sources for {count} cards')
        selected.extend(candidates[:number])
    plan=dict(schema='aic_sfp_diffusion_recovery_pilot/v1',selected=selected,
              actor_prefix_s=6,teacher_cap_s=60,cap=10,
              gate='at least 8/10 full teacher recoveries without scored contact and <10 N sampled cable-board point force',
              source_split='train',diffusion_checkpoint=str(a.diffusion.resolve()),
              separate_sim_gpu=a.gpu,policy_gpu=a.policy_gpu,
              delayed_tf_listener=a.delay_tf,minimum_actor_proposals=20 if a.delay_tf else None,
              actor_inference_p95_limit_ms=300 if a.delay_tf else None,
              privileged_teacher_after_takeover=True,
              final_scenes_opened=False)
    plan_path=a.output/'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text())!=plan:raise ValueError('Frozen pilot plan changed')
    else:plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    progress_path=a.output/'progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
    for row in selected[:a.limit]:
        name=row['name']
        if name in progress:continue
        audit=a.output/'rollouts'/name
        if audit.exists():raise FileExistsError(f'Partial rollout requires manual inspection: {audit}')
        env=os.environ.copy();env.update(
            DOCKER_HOST=f'unix:///run/user/{os.getuid()}/docker.sock',
            AIC_TRANSPORT_EVAL_GPU=str(a.gpu),AIC_TRANSPORT_EVAL_POLICY_GPU=str(a.policy_gpu),
            AIC_TRANSPORT_EVAL_SCENE_DIR=row['scene_dir'],
            AIC_TRANSPORT_EVAL_POLICY='aic_example_policies.ros.RunTemporalThenCheatCodeDiagnostic',
            AIC_TRANSPORT_EVAL_TEACHER_PREFIX='true',
            AIC_TEMPORAL_TRANSPORT_CHECKPOINT=str(TEMPORAL),
            AIC_TEMPORAL_TRANSPORT_IMAGENET=str(IMAGENET),
            AIC_ROUTE_DIFFUSION_CHECKPOINT=str(a.diffusion.resolve()),
            AIC_SFP_CORRECTIVE_ACTOR_DURATION_S='6',
            AIC_SFP_CORRECTIVE_TEACHER_DURATION_S='60',
            AIC_SFP_CORRECTIVE_REQUIRE_FULL_INSERT='1',
            AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA='.06',
            AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC='90',
            AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC='300',
            AIC_TRANSPORT_EVAL_ORACLE_CONTACTS='true',
            AIC_DELAY_TF_LISTENER='1' if a.delay_tf else '0',
            AIC_SFP_DIAGNOSTIC_STOP='0')
        command=['bash','scripts/run_clean_transport_eval.sh',name,str(BASE),str(audit)]
        with (a.output/(name+'.launch.log')).open('w') as log:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,
                                  stderr=subprocess.STDOUT,timeout=750)
        event_path=audit/'corrective_teacher_takeover.json'
        event=json.loads(event_path.read_text()) if event_path.exists() else {}
        score_path=audit/'results/scoring.yaml'
        trial=next((v for k,v in yaml.safe_load(score_path.read_text()).items()
                    if k.startswith('trial_')), {}) if score_path.exists() else {}
        tier3=float(trial.get('tier_3',{}).get('score',0))
        contacts=trial.get('tier_2',{}).get('categories',{}).get('contacts',{})
        cable_force=point_force(audit/'oracle_contacts.jsonl')
        proposals=[]
        trace_path=audit/'runtime_trace.jsonl'
        if trace_path.exists():
            with trace_path.open() as stream:
                for line in stream:
                    item=json.loads(line)
                    if item.get('event')=='proposal':proposals.append(item)
        inference_p95=float(np.quantile([r['inference_ms'] for r in proposals],.95)) if proposals else None
        representative=(not a.delay_tf or (len(proposals)>=20 and
                        inference_p95 is not None and inference_p95<300))
        accepted=bool(result.returncode==0 and event.get('actor_prefix_completed')
                      and event.get('takeover_sim_time_s') is not None and tier3>=75
                      and contacts.get('message')=='No contact detected.'
                      and cable_force is not None and cable_force<10 and representative)
        progress[name]=dict(scene=row['source_scene_hash'],card_count=row['card_count'],
            request=row['alternate_request'],audit=str(audit),returncode=result.returncode,
            takeover_sim_time_s=event.get('takeover_sim_time_s'),
            teacher_outcome=event.get('teacher_outcome'),tier3=tier3,
            contact_message=contacts.get('message'),cable_board_peak_point_force_N=cable_force,
            actor_proposals=len(proposals),actor_inference_p95_ms=inference_p95,
            representative_actor_prefix=representative,accepted=accepted,command=command)
        progress_path.write_text(json.dumps(progress,indent=2)+'\n')
        print(json.dumps(dict(name=name,accepted=accepted,tier3=tier3,
              point_force_N=cable_force,actor_proposals=len(proposals),
              inference_p95_ms=inference_p95)),flush=True)
        done=len(progress);passed=sum(v['accepted'] for v in progress.values())
        if passed+(10-done)<8:
            print(json.dumps(dict(event='stopped_early',done=done,passed=passed,
                                  reason='8/10 gate mathematically impossible')),flush=True)
            break
    done=len(progress);passed=sum(v['accepted'] for v in progress.values())
    summary=dict(schema='aic_sfp_diffusion_recovery_pilot/v1',completed=done,
                 accepted=passed,gate_passed=done==10 and passed>=8,
                 causal_command_join_audited=False,training_started=False,
                 final_scenes_opened=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
