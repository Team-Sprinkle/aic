#!/usr/bin/env python3
"""Score shared visual BC by family and axial phase on grouped held-out episodes."""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_shared_port_tcp_bc import (SharedPortTCPActor, Samples, read_rows,
                                      synthetic_sc_near_pose, synthetic_sc_near_tip)
from shared_port_tcp_geometry import matrix_from_pose9


def phase(row):
    z = abs(float(row['pose'][2]))
    return 'near_0_30mm' if z < .03 else 'approach_30_100mm' if z < .1 else 'transport_over_100mm'


def fine_phase(row):
    z = abs(float(row['pose'][2]))
    if z < .005:
        return 'near_0_5mm'
    if z < .01:
        return 'near_5_10mm'
    if z < .03:
        return 'near_10_30mm'
    return phase(row)


def stats(values):
    return {'count': len(values), 'median': float(np.median(values)) if values else None,
            'p95': float(np.quantile(values, .95)) if values else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sfp-labels', type=Path, required=True)
    p.add_argument('--sc-canonical', type=Path, required=True)
    p.add_argument('--sc-supplemental', type=Path, required=True)
    p.add_argument('--sc-extra', type=Path, nargs='+')
    p.add_argument('--sfp-cache', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-per-episode-phase', type=int, default=25)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--oracle-sc-pose-diagnostic', action='store_true',
                   help='Uses noisy ground-truth SC near pose to test an upper bound, never deployment evidence')
    p.add_argument('--oracle-sc-tip-diagnostic', action='store_true',
                   help='Uses noisy ground-truth SC physical plug-tip pose; never deployment evidence')
    p.add_argument('--fine-phases', action='store_true',
                   help='Separately score 0-5, 5-10, and 10-30 mm axial bands')
    args = p.parse_args()
    torch.set_num_threads(8)
    sources = [('sfp', args.sfp_labels), ('sc', args.sc_canonical),
               ('sc', args.sc_supplemental)]
    if args.sc_extra is not None:
        sources.extend(('sc', path) for path in args.sc_extra)
    rows = read_rows(sources)
    validation = [r for r in rows if r['split'] == 'validation']
    groups = defaultdict(list)
    for row in validation:
        groups[(row['family'], row['episode'],
                fine_phase(row) if args.fine_phases else phase(row))].append(row)
    rng = np.random.default_rng(20260924)
    selected = []
    for group in groups.values():
        index = rng.choice(len(group), min(len(group), args.max_per_episode_phase), replace=False)
        selected.extend(group[int(i)] for i in index)
    loader = DataLoader(Samples(selected, args.sfp_cache), batch_size=args.batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    oracle_noise = checkpoint.get('oracle_sc_near_pose_noise_mm')
    oracle_tip_noise = checkpoint.get('oracle_sc_near_tip_noise_mm')
    if (oracle_noise is not None) != args.oracle_sc_pose_diagnostic:
        raise ValueError('Oracle-conditioned checkpoint and explicit diagnostic flag must agree')
    if (oracle_tip_noise is not None) != args.oracle_sc_tip_diagnostic:
        raise ValueError('Physical-tip oracle checkpoint and explicit diagnostic flag must agree')
    model_state = checkpoint['model']
    keys = ('state_mean', 'state_std', 'pose_mean', 'pose_std', 'target_mean', 'target_std')
    if oracle_tip_noise is not None:
        keys += ('tip_mean', 'tip_std')
    model = SharedPortTCPActor({key: model_state[key].numpy() for key in keys},
                               pretrained=False,
                               use_time=bool(checkpoint.get('include_elapsed', False)),
                               use_tip=oracle_tip_noise is not None).cuda().eval()
    model.load_state_dict(model_state, strict=True)
    values = defaultdict(lambda: defaultdict(list))
    episode_sets = defaultdict(set)
    elapsed = []
    offset = 0
    oracle_generator = torch.Generator(device='cuda').manual_seed(20260924)
    with torch.inference_mode():
        for images, state, task, pose, truth, elapsed_s, families, tip in loader:
            images, state, task = images.cuda(non_blocking=True), state.cuda(non_blocking=True), task.cuda(non_blocking=True)
            external_pose = external_mask = None
            if oracle_noise is not None:
                external_pose, external_mask = synthetic_sc_near_pose(
                    pose.cuda(non_blocking=True), families, oracle_noise, oracle_generator)
            tip_input = tip_mask = None
            if oracle_tip_noise is not None:
                tip_input, tip_mask = synthetic_sc_near_tip(
                    tip.cuda(non_blocking=True), pose.cuda(non_blocking=True),
                    families, oracle_tip_noise, oracle_generator)
            torch.cuda.synchronize(); before = time.perf_counter()
            predicted, estimated_pose = model(images, state, task,
                elapsed_s.cuda(non_blocking=True), external_pose, external_mask,
                tip_input, tip_mask)
            torch.cuda.synchronize(); elapsed.append((time.perf_counter() - before)*1000)
            predicted, estimated_pose = predicted.cpu().numpy(), estimated_pose.cpu().numpy()
            conditioned_pose = estimated_pose.copy()
            if external_pose is not None:
                mask = external_mask.cpu().numpy()
                conditioned_pose[mask] = external_pose.cpu().numpy()[mask]
            truth, pose = truth.numpy(), pose.numpy()
            for i in range(len(predicted)):
                row = selected[offset+i]
                group = (row['family'], fine_phase(row) if args.fine_phases else phase(row))
                episode_sets[group].add(row['episode'])
                target_error = float(np.linalg.norm(predicted[i, 0, :3] - truth[i, 0, :3])*1000)
                pose_error = float(np.linalg.norm(estimated_pose[i, :3] - pose[i, :3])*1000)
                predicted_action = np.linalg.inv(matrix_from_pose9(conditioned_pose[i])) @ matrix_from_pose9(predicted[i, 0])
                true_action = np.linalg.inv(matrix_from_pose9(pose[i])) @ matrix_from_pose9(truth[i, 0])
                command_error = float(np.linalg.norm(predicted_action[:3, 3] - true_action[:3, 3])*1000)
                # Both complete targets and current poses are expressed in the
                # selected opening frame. Its z axis is the insertion axis.
                # Score the intended displacement in that frame, independent
                # of TCP-body orientation.
                predicted_port_command = predicted[i, 0, :3] - conditioned_pose[i, :3]
                true_port_command = truth[i, 0, :3] - pose[i, :3]
                port_command_error = (predicted_port_command - true_port_command) * 1000
                values[group]['target_translation_mm'].append(target_error)
                values[group]['pose_translation_mm'].append(pose_error)
                values[group]['body_command_translation_mm'].append(command_error)
                values[group]['lateral_command_mm'].append(float(np.linalg.norm(port_command_error[:2])))
                values[group]['axial_command_mm'].append(float(abs(port_command_error[2])))
                values[group]['lateral_target_mm'].append(float(np.linalg.norm((predicted[i, 0, :2] - truth[i, 0, :2])*1000)))
            offset += len(predicted)
    report = {'schema': 'shared_port_tcp_bc_grouped_offline/v1',
              'checkpoint': str(args.checkpoint), 'checkpoint_update': checkpoint['update'],
              'runtime_inputs': 'three RGB 224x224 views, ordinary 32D state, 10D task; pose head prediction',
              'truth_used_during_inference': bool(args.oracle_sc_pose_diagnostic or
                                                  args.oracle_sc_tip_diagnostic),
              'oracle_sc_near_pose_noise_mm': oracle_noise,
              'oracle_sc_near_tip_noise_mm': oracle_tip_noise,
              'fine_phases': bool(args.fine_phases),
              'phase_selection_uses_training_label_only': True,
              'scenes': len({r['scene'] for r in selected}),
              'validation_rows_sampled': len(selected),
              'by_family_phase': {f'{family}/{ph}': {'episodes': len(episode_sets[(family, ph)]),
                  **{key: stats(samples) for key, samples in metrics.items()}}
                  for (family, ph), metrics in values.items()},
              'batched_model_only_latency_ms': stats(elapsed),
              'latency_note': 'Batch forward only; excludes camera decode, ROS, high-resolution SC perception, conversion and control'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'scenes': report['scenes'], 'rows': len(selected),
                      'by_family_phase': report['by_family_phase']}))


if __name__ == '__main__':
    main()
