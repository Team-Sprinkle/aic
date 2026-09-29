#!/usr/bin/env python3
"""Assemble scored new SC successes without admitting failed routes to BC."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    episodes = []; ports = []; calibrations = []; sources = []
    seen = set()
    for run in args.runs:
        edge_file = run / 'edges_joined.json'
        calibration_file = run / 'dynamic_grasp_calibration.json'
        edges = json.loads(edge_file.read_text())['rows']
        calibration = {x['trial']: x for x in json.loads(calibration_file.read_text())['episodes']}
        sources.append({'run': str(run), 'edges': str(edge_file),
                        'edges_sha256': sha256(edge_file),
                        'grasp_calibration': str(calibration_file),
                        'grasp_calibration_sha256': sha256(calibration_file)})
        for item in edges:
            trial = item['trial']
            if trial in seen:
                raise ValueError(f'Duplicate trial {trial}')
            seen.add(trial)
            if abs(float(item['official_tier3']) - 75) > 1e-6:
                continue
            if trial not in calibration:
                raise ValueError(f'No physical grasp calibration for {trial}')
            drift = max(x['translation_change_from_first_mm']
                        for x in calibration[trial]['samples'])
            if drift > .25:
                raise ValueError(f'Physical grasp is not sufficiently rigid: {trial} {drift}')
            folder = Path(item['episode'])
            frames_file = folder / 'frames.jsonl'
            frames = [json.loads(x) for x in frames_file.open()]
            times = np.array([float(x['sim_time']) for x in frames])
            indices = np.array([int(x['command_index']) for x in frames])
            if not np.all(np.diff(times) > 0) or not np.all(np.diff(indices) >= 0):
                raise ValueError(f'Noncausal frame/command sequence: {trial}')
            if any(set(x.get('images', {})) != {'center', 'left', 'right'} or
                   not all((folder / name).is_file() for name in x['images'].values())
                   for x in frames):
                raise ValueError(f'Missing RGB in {trial}')
            if any(np.max(np.abs(np.asarray(x['teacher_target_pose']) -
                                 np.asarray(x['executed_target_pose']))) > 1e-9
                   or x.get('student_active') or
                   any(abs(float(v)) > 1e-12 for v in x.get('perturbation_xyz_rotvec', []))
                   for x in frames):
                raise ValueError(f'Not a clean expert target stream: {trial}')
            port = int(item['target_module_name'].rsplit('_', 1)[-1])
            index = int(trial.removeprefix('trial_'))
            task_vector = [0., 1., *([1., 0.] if port == 0 else [0., 1.]), *([0.] * 6)]
            episodes.append({'episode_index': index, 'trial_id': trial,
                             'split': 'train', 'scene_sha256': item['scene_sha256'],
                             'canonical_episode_dir': str(folder),
                             'task': {'task_family': 'sc_to_sc', 'target_port_index': port},
                             'task_vector': task_vector,
                             'official_tier3': item['official_tier3'],
                             'frames': len(frames), 'frames_sha256': sha256(frames_file),
                             'native_image_frames': sum(bool(x.get('native_images')) for x in frames),
                             'max_measured_grasp_drift_mm': drift,
                             'source_run': str(run),
                             'status': 'scored_clean_candidate; causal chunk admission pending'})
            ports.append({'episode_index': index, 'target_port_index': port,
                          'scene_sha256': item['scene_sha256'], 'edges': item['edges']})
            calibrations.append(calibration[trial])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'schema': 'sc_supplemental_scored_candidates/v1',
                'training_label_only': True, 'bc_admitted': False,
                'sources': sources, 'episodes': episodes}
    port_report = {'schema': 'sc_supplemental_port_edges/v1', 'episodes': ports}
    tip_report = {'schema': 'sc_supplemental_dynamic_grasp/v1',
                  'training_only': True, 'episodes': calibrations}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output_dir / 'port_edges.json').write_text(json.dumps(port_report, indent=2) + '\n')
    (args.output_dir / 'physical_grasp_calibrations.json').write_text(json.dumps(tip_report, indent=2) + '\n')
    print(json.dumps({'successful_candidates': len(episodes),
                      'by_port': {str(port): sum(e['task']['target_port_index'] == port for e in episodes)
                                  for port in (0, 1)}}))


if __name__ == '__main__':
    main()
