#!/usr/bin/env python3
"""Admit scored SC teacher successes after causal-label and motion checks.

This preserves the original candidate manifest and every rejected rollout.
The force/stall screen catches sustained unsafe pushing, not ordinary 20 N
tool-weight bias or momentary successful insertion contact.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate-manifest', type=Path, required=True)
    p.add_argument('--waypoint-labels', type=Path, required=True)
    p.add_argument('--verified-manifest', type=Path, required=True)
    p.add_argument('--existing-labels', type=Path, nargs='*', default=[],
                   help='Every SC label source already used for BC; reject any duplicate scene or episode')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    candidates = json.loads(a.candidate_manifest.read_text())
    verified = json.loads(a.verified_manifest.read_text())['episodes']
    existing = {r['scene_sha256'] for r in verified}
    existing_episode_ids = set()
    for source in a.existing_labels:
        for line in source.open():
            row = json.loads(line)
            existing.add(row['scene_sha256'])
            existing_episode_ids.add(int(row['episode_index']))
    episodes = candidates['episodes']
    if len({r['scene_sha256'] for r in episodes}) != len(episodes):
        raise ValueError('Repeated supplemental scene')
    if any(r['scene_sha256'] in existing for r in episodes):
        raise ValueError('Supplemental scene overlaps verified train/validation')
    if any(int(r['episode_index']) in existing_episode_ids for r in episodes):
        raise ValueError('Supplemental episode ID overlaps an existing BC source')
    labels = defaultdict(list)
    for line in a.waypoint_labels.open():
        row = json.loads(line)
        labels[int(row['episode_index'])].append(row)
    if set(labels) != {int(e['episode_index']) for e in episodes}:
        raise ValueError('Episode/label identity mismatch')
    audited = []
    for episode in episodes:
        index = int(episode['episode_index'])
        rows = labels[index]
        if not rows or not all(r['scene_sha256'] == episode['scene_sha256'] for r in rows):
            raise ValueError(f'Label scene mismatch: {index}')
        if not all(r['tcp_tip_label_source'] == 'episode_scored_physical_tf' for r in rows):
            raise ValueError(f'Fixed-proxy label in physical dataset: {index}')
        if any(len(r['teacher_sc_tip_target_poses_opening_frame']) != 4 or
               set(r['images']) != {'center', 'left', 'right'} for r in rows):
            raise ValueError(f'Incomplete target or image label: {index}')
        times = np.asarray([float(r['sim_time']) for r in rows])
        if not np.all(np.diff(times) > 0):
            raise ValueError(f'Noncausal target chunks: {index}')
        force = np.asarray([np.linalg.norm(r['state'][26:29]) for r in rows])
        speed_mm_s = np.asarray([np.linalg.norm(r['state'][7:10]) * 1000 for r in rows])
        gap_mm = np.asarray([np.linalg.norm(
            np.asarray(r['teacher_sc_tip_target_poses_opening_frame'][0][:3]) -
            np.asarray(r['observed_sc_tip_pose_opening_frame'][:3])) * 1000 for r in rows])
        baseline_force = float(np.median(force[:min(100, len(force))]))
        severe = (force > baseline_force + 10) & (speed_mm_s < 1) & (gap_mm > 20)
        full_score = abs(float(episode['official_tier3']) - 75) < 1e-6
        passed = bool(full_score and episode['split'] == 'train' and
                      episode['status'].startswith('scored_clean_candidate') and not severe.any())
        audited.append({'episode_index': index, 'trial_id': episode['trial_id'],
                        'scene_sha256': episode['scene_sha256'],
                        'score_tier3': episode['official_tier3'],
                        'physical_grasp_drift_mm': episode['max_measured_grasp_drift_mm'],
                        'causal_chunks': len(rows), 'baseline_force_n': baseline_force,
                        'force_above_baseline_plus_10_n_chunks': int((force > baseline_force + 10).sum()),
                        'severe_force_stall_chunks': int(severe.sum()),
                        'teacher_setpoint_gap_p95_mm': float(np.percentile(gap_mm, 95)),
                        'teacher_setpoint_gap_max_mm': float(gap_mm.max()),
                        'admitted': passed})
    output = {'schema': 'sc_bc_admission/v1',
              'candidate_manifest': str(a.candidate_manifest),
              'candidate_manifest_sha256': digest(a.candidate_manifest),
              'waypoint_labels': str(a.waypoint_labels),
              'waypoint_labels_sha256': digest(a.waypoint_labels),
              'verified_manifest': str(a.verified_manifest),
              'verified_manifest_sha256': digest(a.verified_manifest),
              'existing_labels': [{'path': str(source), 'sha256': digest(source)}
                                  for source in a.existing_labels],
              'all_episode_splits': 'train; no scene overlaps verified data',
              'training_label_only_geometry': True,
              'admitted_episodes': sum(r['admitted'] for r in audited),
              'admitted_chunks': sum(r['causal_chunks'] for r in audited if r['admitted']),
              'episodes': audited,
              'caveat': 'Official success and a force/stall screen support BC admission. Large teacher setpoint lookahead is retained and must be validated by autonomous rollout; these are not recovery demonstrations.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'episodes': output['admitted_episodes'],
                      'chunks': output['admitted_chunks']}))


if __name__ == '__main__':
    main()
