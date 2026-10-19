#!/usr/bin/env python3
"""Three train-only actor-to-teacher takeovers; inspect measured recovery paths."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from run_sfp_cheatcode_target_pilot import BASE, ROOT, score
from run_sfp_paired_actor_live_three import extract_teacher

SOURCE=Path('/home/nas/chmin/aic_archives/2026-10-07_sfp_divergent_pairs_round2_v1')
CHECKPOINT=Path('/var/tmp/chmin_aic_20260918_act/sfp_history_context_live_20261007/corrected.pt')
RESNET=Path('/var/tmp/aic_sc_fix_20260924/resnet18-f37072fd.pth')
SCENES=(
    'episode2400089_card1p0_to_card4p1',  # two cards, port 1
    'episode2400020_card2p1_to_card0p0',  # three cards, port 0
    'episode2400110_card4p1_to_card0p0',  # five cards, port 0
)


def measured_path(audit, takeover_s):
    data=json.loads((audit/'scored_stream.json').read_text())
    stream=data['streams']['/aic_controller/controller_state']
    times=np.asarray([row[0]*1e-9 for row in stream],float)
    xyz=np.asarray([[row[1]['tcp_pose']['position'][axis] for axis in ('x','y','z')]
                    for row in stream],float)
    quat=np.asarray([[row[1]['tcp_pose']['orientation'][axis] for axis in ('x','y','z','w')]
                     for row in stream],float)
    k=int(np.searchsorted(times,takeover_s))
    if k>=len(times):return dict(valid=False,reason='no_terminal_tcp_after_takeover')
    # One possible hold command is issued at takeover; use the first measured
    # TCP state after it as the origin, while retaining its exact timestamp.
    path=0.;chosen=None
    for j in range(k+1,len(times)):
        if times[j]-times[k]>4:break
        path+=float(np.linalg.norm(xyz[j]-xyz[j-1]))
        if path>=.020:
            chosen=j;break
    if chosen is None:return dict(valid=False,reason='less_than_20mm_path_within_4s',
                                  observed_path_mm=path*1000,
                                  observed_time_s=float(min(times[-1]-times[k],4)))
    body=Rotation.from_quat(quat[k]).inv().apply(xyz[chosen]-xyz[k])
    net=float(np.linalg.norm(body)*1000)
    return dict(valid=net>=10,reason='valid' if net>=10 else 'net_below_10mm',
                origin_sim_s=float(times[k]),lookahead_s=float(times[chosen]-times[k]),
                path_mm=path*1000,net_mm=net,direction_tcp=(body/np.linalg.norm(body)).tolist(),
                displacement_tcp_mm=(body*1000).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=7)
    p.add_argument('--actor-prefix-s',type=int,default=5)
    a=p.parse_args()
    if not 1<=a.actor_prefix_s<=20 or not 0<=a.gpu<=7:p.error('Invalid bound')
    if a.output.exists():p.error('Preserve prior pilot output')
    plan=json.loads((SOURCE/'plan.json').read_text())
    progress=json.loads((SOURCE/'progress.json').read_text())['results']
    if not (plan.get('train_only') or plan.get('source_episode_grouped')) or plan.get('final_scenes_opened'):
        p.error('Source must be sealed and episode-grouped')
    source={s['name']:s for s in plan['scenes']}
    chosen=[source[name] for name in SCENES]
    if any(s.get('source_split','train')!='train' for s in chosen):
        p.error('Pilot source must use only training reset groups')
    if any(progress[s['name']]['official_score'].get('tier3',0)<75 for s in chosen):
        p.error('All source teachers must have succeeded')
    existing=json.loads((ROOT/'outputs/experiments/2026-10-08_sfp_short_waypoint_feasibility_v1/arc20_labels/manifest.json').read_text())
    old={x['scene'] for x in existing['actor_boundary']}
    if any(s['source_scene_hash'] in old for s in chosen):p.error('Pilot scene overlaps prior actor-boundary labels')
    a.output.mkdir(parents=True)
    manifest=dict(schema='aic_sfp_actor_recovery_label_pilot/v1',source_plan=str(SOURCE/'plan.json'),
                  checkpoint=str(CHECKPOINT),actor_prefix_s=a.actor_prefix_s,
                  selected=[dict(name=s['name'],scene=s['source_scene_hash'],card_count=s['card_count'],
                                 request=s['alternate_request']) for s in chosen],
                  train_only=True,final_scenes_opened=False,maximum_runs=3,
                  decision_gate='At least 2/3 safe full teacher insertions and valid 20mm path labels')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    results=[]
    for spec in chosen:
        name=spec['name']+f'_actor{a.actor_prefix_s}s_teacher_full'
        audit=a.output/'rollouts'/name
        env=os.environ.copy()
        env.update(DOCKER_HOST=f'unix:///run/user/{os.getuid()}/docker.sock',
                   AIC_TRANSPORT_EVAL_GPU=str(a.gpu),AIC_TRANSPORT_EVAL_POLICY_GPU=str(a.gpu),
                   AIC_TRANSPORT_EVAL_SCENE_DIR=spec['scene_dir'],
                   AIC_TRANSPORT_EVAL_POLICY='aic_example_policies.ros.RunTemporalThenCheatCodeDiagnostic',
                   AIC_TRANSPORT_EVAL_TEACHER_PREFIX='true',
                   AIC_TEMPORAL_TRANSPORT_CHECKPOINT=str(CHECKPOINT),
                   AIC_TEMPORAL_TRANSPORT_IMAGENET=str(RESNET),
                   AIC_SFP_CORRECTIVE_ACTOR_DURATION_S=str(a.actor_prefix_s),
                   AIC_SFP_CORRECTIVE_TEACHER_DURATION_S='60',
                   AIC_SFP_CORRECTIVE_REQUIRE_FULL_INSERT='1',
                   AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA='.06',
                   AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC='90',
                   AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC='300',
                   AIC_SFP_DIAGNOSTIC_STOP='0')
        command=['bash','scripts/run_clean_transport_eval.sh',name,str(BASE),str(audit)]
        (a.output/(name+'.command.json')).write_text(json.dumps(dict(command=command,
            environment={k:v for k,v in env.items() if k.startswith('AIC_') or k=='DOCKER_HOST'}),indent=2)+'\n')
        with (a.output/(name+'.launch.log')).open('w') as log:
            try:rc=subprocess.run(command,cwd=ROOT,env=env,stdout=log,
                                  stderr=subprocess.STDOUT,timeout=500).returncode
            except subprocess.TimeoutExpired:rc=None
        event_path=audit/'corrective_teacher_takeover.json'
        event=json.loads(event_path.read_text()) if event_path.exists() else {}
        official=score(audit)
        contact=None
        scoring=audit/'results/scoring.yaml'
        if scoring.exists():
            trials=yaml.safe_load(scoring.read_text())
            trial=next((v for k,v in trials.items() if k.startswith('trial_')), {})
            contact=float(trial.get('tier_2',{}).get('categories',{}).get('contacts',{}).get('score',0))
        if rc==0 and event.get('takeover_sim_time_s') and scoring.exists():
            extract_teacher(audit)
            path=measured_path(audit,float(event['takeover_sim_time_s']))
        else:path=dict(valid=False,reason='invalid_rollout_or_missing_takeover')
        row=dict(name=name,scene=spec['source_scene_hash'],request=spec['alternate_request'],
                 card_count=spec['card_count'],audit=str(audit),returncode=rc,
                 official_score=official,contact_score=contact,
                 teacher_outcome=event.get('teacher_outcome'),path=path,
                 safe_full_insert=bool(rc==0 and official.get('tier3',0)>=75 and contact is not None and contact>=0),
                 valid_label=bool(path['valid']))
        results.append(row)
        (a.output/'summary.json').write_text(json.dumps(dict(schema=manifest['schema'],runs=results,
            completed=len(results),safe_full_insertions=sum(x['safe_full_insert'] for x in results),
            valid_labels=sum(x['valid_label'] for x in results),
            pilot_gate_passed=(len(results)==3 and sum(x['safe_full_insert'] and x['valid_label'] for x in results)>=2),
            final_scenes_opened=False),indent=2)+'\n')
        print(json.dumps(row),flush=True)
        if rc is None:raise RuntimeError(f'Timed out: inspect {audit}')

if __name__=='__main__':main()
