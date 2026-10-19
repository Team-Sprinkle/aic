#!/usr/bin/env python3
"""Wait for both bounded captures, then train a matched near-view pose model."""
import json
import os
from pathlib import Path
import subprocess
import time

repo = Path.cwd()
python = repo/'.pixi/envs/default/bin/python'
root = Path('/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1')
state = root/'near_pose_repair_state.json'
if state.exists():
    raise FileExistsError('Do not duplicate close-view repair')


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save(data):
    temporary = state.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(state)


save(dict(stage='waiting_for_far_dataset_and_12_verified_near_holds'))
while not (read(root/'acquisition_pose_dataset/summary.json') and read(root/'near25_captures/summary.json')):
    time.sleep(15)
assert read(root/'near25_captures/summary.json')['completed'] == 12
source = root/'far_and_near_pose_dataset'
command = [python, repo/'scripts/merge_sc_grasp_capture_supervision.py',
           '--source', root/'acquisition_pose_dataset', '--captures', root/'near25_captures',
           '--output', source]
(root/'near_dataset.command.json').write_text(json.dumps(dict(command=list(map(str, command))), indent=2)+'\n')
save(dict(stage='merging_immutable_far_and_near_labels'))
with (root/'near_dataset.log').open('w') as log:
    subprocess.run(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT, check=True)
command = [python, repo/'scripts/train_sc_spatial_grasp_to_plateau.py', '--source', source,
           '--pixels', source/'native_crops.npy', '--checkpoint',
           '/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/sc_grasp_joint_geometry_v6_continuation/joint_geometry/best.pt',
           '--output', root/'far_near_spatial_joint', '--mode', 'joint', '--crop-jitter', '0',
           '--selection-domains', 'original', 'nominal_acquisition', 'near_25mm',
           '--updates', '100000', '--minutes', '1440', '--min-updates', '5000', '--patience', '3000']
(root/'far_near_spatial_joint.command.json').write_text(json.dumps(dict(command=list(map(str, command)),
    physical_gpu=3, stopping_rule='LR-floor validation plateau; a time/update ceiling is incomplete',
    deployment_gate='Every selected domain translationp95<=0.4mm,orientationp95<=1degree. Report tilt/twist and tighter SC seating sensitivity separately.',
    paired_ablation='Same architecture/init/optimizer as far-only model, adding12 complete near-view episodes.'), indent=2)+'\n')
save(dict(stage='training_far_plus_near_pose_to_plateau', physical_gpu=3))
env = dict(os.environ, CUDA_VISIBLE_DEVICES='3', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
with (root/'far_near_spatial_joint.log').open('w') as log:
    subprocess.run(list(map(str, command)), env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
report = read(root/'far_near_spatial_joint/summary.json')
if not report['plateau_reached']:
    raise RuntimeError('Incomplete fit; resume exact checkpoint')
save(dict(stage='near_pose_fit_complete_requires_prediction_QA_and_live_test',
          gate_passed=report['gate_passed'], metrics=report['metrics_by_domain'],
          automatic_model_swap=False))
