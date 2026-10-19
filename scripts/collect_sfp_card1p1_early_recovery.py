#!/usr/bin/env python3
"""Bounded early takeover on train-only five-card card1/port1 resets."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import yaml

from collect_sfp_diffusion_recovery_pilot import point_force, BASE, TEMPORAL, IMAGENET, ROOT


SOURCE = Path('/var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/sfp_divergent_merged_round2_v1/sfp_labels.jsonl')
TEMPLATE = Path('/home/nas/chmin/aic_archives/2026-10-05_sfp_paired_coverage_v2/scenes/card0_port0_trial_000421_to_card1_port1')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_plan(output, diffusion):
    seen = {}
    with SOURCE.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['split'] == 'train' and row.get('source_scene_yaml'):
                seen.setdefault(int(row['episode_index']), row)
    eligible = []
    for episode, row in sorted(seen.items()):
        source = Path(row['source_scene_yaml'])
        config = yaml.safe_load(source.read_text())
        trial = next(iter(config['trials'].values()))
        board = trial['scene']['task_board']
        if all(board[f'nic_rail_{card}']['entity_present'] for card in range(5)):
            eligible.append((episode, row, source, config))
    if len(eligible) < 10:
        raise ValueError(f'Only {len(eligible)} five-card train resets')
    chosen = [eligible[i] for i in np.linspace(0, len(eligible)-1, 10).astype(int)]
    if len({row['scene_sha256'] for _, row, _, _ in chosen}) != 10:
        raise ValueError('Selected scenes are not distinct reset groups')
    scenes = []
    for episode, row, source, config in chosen:
        trial = next(iter(config['trials'].values()))
        task = next(iter(trial['tasks'].values()))
        task['target_module_name'] = 'nic_card_mount_1'
        task['port_name'] = 'sfp_port_1'
        name = f'train_episode{episode}_fivecard_card1p1'
        folder = output/'scenes'/name
        folder.mkdir(parents=True)
        (folder/'eval_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        for file in ('world_audit.sdf', 'bridge_audit.yaml'):
            shutil.copy2(TEMPLATE/file, folder/file)
        scenes.append(dict(name=name,source_episode=episode,source_split='train',
                           source_scene_hash=row['scene_sha256'],source_trial=str(source),
                           card_count=5,alternate_request=[1,1],scene_dir=str(folder),
                           config_sha256=sha(folder/'eval_config.yaml')))
    plan=dict(schema='aic_sfp_card1p1_early_recovery/v1',scenes=scenes,
              source_labels=str(SOURCE),source_labels_sha256=sha(SOURCE),
              actor_checkpoint=str(diffusion.resolve()),actor_prefix_s=4,
              teacher_cap_s=60,first_three_gate='at least 2/3 accepted',
              total_gate='at least 8/10 accepted',
              acceptance='full insert, no scored contact, cable-board point force <10N, '
                         'at least 12 actor proposals, actor p95 <300ms',
              source_episode_grouped=True,final_scenes_opened=False,
              privileged_teacher_after_takeover=True)
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diffusion',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=3)
    parser.add_argument('--policy-gpu',type=int,default=4)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    plan_path=args.output/'plan.json'
    plan=(json.loads(plan_path.read_text()) if plan_path.exists()
          else make_plan(args.output,args.diffusion))
    if plan['final_scenes_opened'] or plan['actor_checkpoint']!=str(args.diffusion.resolve()):
        parser.error('Frozen plan mismatch')
    progress_path=args.output/'progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
    for index,scene in enumerate(plan['scenes']):
        if index==3 and sum(v['accepted'] for v in progress.values())<2:
            print(json.dumps(dict(event='first_three_gate_failed',completed=len(progress))),flush=True)
            break
        name=scene['name']
        if name in progress:continue
        folder=Path(scene['scene_dir'])
        if sha(folder/'eval_config.yaml')!=scene['config_sha256']:
            raise ValueError('Scene changed')
        audit=args.output/'rollouts'/name
        if audit.exists():raise FileExistsError(f'Inspect partial rollout: {audit}')
        env=os.environ.copy();env.update(
            DOCKER_HOST=f'unix:///run/user/{os.getuid()}/docker.sock',
            AIC_TRANSPORT_EVAL_GPU=str(args.gpu),AIC_TRANSPORT_EVAL_POLICY_GPU=str(args.policy_gpu),
            AIC_TRANSPORT_EVAL_SCENE_DIR=str(folder),
            AIC_TRANSPORT_EVAL_POLICY='aic_example_policies.ros.RunTemporalThenCheatCodeDiagnostic',
            AIC_TRANSPORT_EVAL_TEACHER_PREFIX='true',AIC_DELAY_TF_LISTENER='1',
            AIC_TEMPORAL_TRANSPORT_CHECKPOINT=str(TEMPORAL),
            AIC_TEMPORAL_TRANSPORT_IMAGENET=str(IMAGENET),
            AIC_ROUTE_DIFFUSION_CHECKPOINT=str(args.diffusion.resolve()),
            AIC_SFP_CORRECTIVE_ACTOR_DURATION_S='4',
            AIC_SFP_CORRECTIVE_TEACHER_DURATION_S='60',
            AIC_SFP_CORRECTIVE_REQUIRE_FULL_INSERT='1',
            AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA='.06',
            AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC='90',
            AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC='300',
            AIC_TRANSPORT_EVAL_ORACLE_CONTACTS='true',AIC_SFP_DIAGNOSTIC_STOP='0')
        command=['bash','scripts/run_clean_transport_eval.sh',name,str(BASE),str(audit)]
        with (args.output/(name+'.launch.log')).open('w') as log:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,
                                  stderr=subprocess.STDOUT,timeout=750)
        event_path=audit/'corrective_teacher_takeover.json'
        event=json.loads(event_path.read_text()) if event_path.exists() else {}
        score_path=audit/'results/scoring.yaml'
        official=yaml.safe_load(score_path.read_text()) if score_path.exists() else {}
        trial=next((v for k,v in official.items() if k.startswith('trial_')), {})
        tier3=float(trial.get('tier_3',{}).get('score',0))
        contacts=trial.get('tier_2',{}).get('categories',{}).get('contacts',{})
        cable_force=point_force(audit/'oracle_contacts.jsonl')
        proposals=[json.loads(line) for line in (audit/'runtime_trace.jsonl').open()
                   if '"event": "proposal"' in line] if (audit/'runtime_trace.jsonl').exists() else []
        inference_p95=(float(np.quantile([r['inference_ms'] for r in proposals],.95))
                       if proposals else None)
        accepted=bool(result.returncode==0 and event.get('actor_prefix_completed')
                      and event.get('takeover_sim_time_s') is not None and tier3>=75
                      and contacts.get('message')=='No contact detected.'
                      and cable_force is not None and cable_force<10
                      and len(proposals)>=12 and inference_p95 is not None and inference_p95<300)
        progress[name]=dict(scene=scene['source_scene_hash'],card_count=5,
            request=[1,1],audit=str(audit),returncode=result.returncode,
            takeover_sim_time_s=event.get('takeover_sim_time_s'),
            tier3=tier3,contact_message=contacts.get('message'),
            cable_board_peak_point_force_N=cable_force,actor_proposals=len(proposals),
            actor_inference_p95_ms=inference_p95,accepted=accepted,command=command)
        progress_path.write_text(json.dumps(progress,indent=2)+'\n')
        print(json.dumps(dict(name=name,accepted=accepted,tier3=tier3,
                              force_N=cable_force,proposals=len(proposals),
                              inference_p95_ms=inference_p95)),flush=True)
        if sum(v['accepted'] for v in progress.values())+10-len(progress)<8:
            print(json.dumps(dict(event='total_gate_impossible',completed=len(progress))),flush=True)
            break
    done=len(progress);accepted=sum(v['accepted'] for v in progress.values())
    summary=dict(schema='aic_sfp_diffusion_recovery_pilot/v1',completed=done,
                 accepted=accepted,gate_passed=done==10 and accepted>=8,
                 first_three_gate_passed=(sum(progress[x['name']]['accepted']
                                              for x in plan['scenes'][:3] if x['name'] in progress)>=2),
                 causal_command_join_audited=False,training_started=False,
                 final_scenes_opened=False)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
