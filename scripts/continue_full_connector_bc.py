#!/usr/bin/env python3
"""One bounded full-object representation BC diagnostic after data audits.

No automatic promotion, broad sweep or RL. Frozen legacy comparison is kept.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();repo=Path.cwd();state=a.root/'continuation_state.json'
    save(state,dict(stage='waiting_for_RGB_only_full_pose_derivative',new_RL=False))
    while not (a.root/'full_connector_actor_rows/summary.json').exists():
        command=json.loads((a.root/'build_full_connector_rows.command.json').read_text())
        try:os.kill(command['pid'],0)
        except ProcessLookupError:
            save(state,dict(stage='data_preparation_failed_requires_diagnosis',new_RL=False));return
        time.sleep(20)
    for name in ['orientation_actor_split_audit.json','full_actor_decode_preflight.json']:
        if not json.loads((a.root/name).read_text())['passed']:raise ValueError(name+' failed')
    dataset=json.loads((a.root/'full_connector_actor_rows/summary.json').read_text())
    if dataset['rows']!=60671 or not dataset['original_point_targets_unchanged']:
        raise ValueError('Unmatched anchors or changed recorded command positions')
    baseline=Path('/home/nas/chmin/aic_archives/2026-10-17_history_chunk_continuation_v1/h4.command.json')
    command=json.loads(baseline.read_text())['command']
    command[1]=str(repo/'scripts/train_full_connector_actor.py')
    for name in ['--action-horizon','--overlap-consistency-weight']:
        i=command.index(name);del command[i:i+2]
    for name,value in [('--tip-input-mode','full_pose'),('--sc-action-reference','full_connector_pose'),
                       ('--dataset',str(a.root/'full_connector_actor_rows')),('--output',str(a.root/'full_connector_bc_h4'))]:
        command[command.index(name)+1]=value
    save(a.root/'full_connector_bc.command.json',dict(command=command,physical_gpu=4,
        comparison='Same H4 architecture, anchors, images, split, history and convergence opportunity as prior h4; explicit full connector input/output replaces hybrid geometry.',
        stopping_rule='>=10k updates plus60 stale validations;50k/6h cap explicitly distinguished from plateau.',
        live_gate='No claim of improvement until fixed development rollouts with image-only stable grasp rotation. No final-scene access.',
        teacher_phase_ambiguity_not_fixed_by_this_arm=True))
    save(state,dict(stage='training_full_connector_bc_h4',physical_gpu=4,new_RL=False,
        actor_deployment_approved=False))
    with (a.root/'full_connector_bc.log').open('w') as log:
        result=subprocess.run(command,cwd=repo,env=dict(os.environ,CUDA_VISIBLE_DEVICES='4'),
            stdout=log,stderr=subprocess.STDOUT)
    summary=a.root/'full_connector_bc_h4/summary.json'
    save(state,dict(stage='BC_fit_finished_requires_runtime_integration' if result.returncode==0 and summary.exists()
        else 'BC_fit_failed_requires_diagnosis',returncode=result.returncode,summary=str(summary),
        next='Validate full-pose runtime acquisition/decoding, then frozen no-recovery development rollouts. Stop this policy branch if geometry improves but alignment does not.',
        actor_deployment_approved=False,new_RL=False,final_scenes_opened=False))


if __name__=='__main__':main()
