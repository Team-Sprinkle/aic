#!/usr/bin/env python3
"""Bounded acquisition-view repair; wait for QA, merge, fit to LR-floor plateau."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    state = root/'pose_repair_state.json'
    if state.exists():
        raise FileExistsError('Inspect existing continuation; do not duplicate')
    repo = Path.cwd()
    python = repo/'.pixi/envs/default/bin/python'
    save(state, dict(stage='waiting_for_36_QA_verified_acquisition_holds'))
    while not read(root/'acquisition_captures/summary.json'):
        time.sleep(15)
    capture = read(root/'acquisition_captures/summary.json')
    assert capture['completed'] == 36 and all(row['valid'] for row in capture['results'])
    source = root/'acquisition_pose_dataset'
    command = [python, repo/'scripts/merge_sc_grasp_capture_supervision.py',
               '--source', '/home/nas/chmin/aic_archives/2026-10-17_insertion_control_repair_v1/sc_nominal_coverage_v3',
               '--captures', root/'acquisition_captures', '--output', source]
    save(root/'pose_dataset.command.json', dict(command=list(map(str, command))))
    save(state, dict(stage='merging_immutable_perception_sources'))
    with (root/'pose_dataset.log').open('w') as log:
        subprocess.run(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT, check=True)
    command = [python, repo/'scripts/train_sc_spatial_grasp_to_plateau.py',
               '--source', source, '--pixels', source/'native_crops.npy',
               '--checkpoint', '/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/sc_grasp_joint_geometry_v6_continuation/joint_geometry/best.pt',
               '--output', root/'acquisition_spatial_joint', '--mode', 'joint',
               '--crop-jitter', '0', '--selection-domains', 'original', 'nominal_acquisition',
               '--updates', '100000', '--minutes', '1440', '--min-updates', '5000', '--patience', '3000']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='4', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    save(root/'acquisition_spatial_joint.command.json', dict(command=list(map(str, command)),
         physical_gpu=4, stopping_rule='Validation plateau at LR floor; time/update cap is incomplete',
         deployment_domains=['original', 'nominal_acquisition'], home_metrics_still_reported=True))
    save(state, dict(stage='training_spatial_joint_to_validation_plateau', physical_gpu=4))
    with (root/'acquisition_spatial_joint.log').open('w') as log:
        subprocess.run(list(map(str, command)), env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = read(root/'acquisition_spatial_joint/summary.json')
    if not report['plateau_reached']:
        raise RuntimeError('Fit incomplete; resume exact checkpoint before judging it')
    save(state, dict(stage='pose_fit_complete_requires_prediction_QA_and_live_test',
         gate_passed=report['gate_passed'], metrics=report['metrics_by_domain'],
         automatic_model_swap=False))


if __name__ == '__main__':
    main()
