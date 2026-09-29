#!/usr/bin/env python3
"""Audit newly scored SC teacher episodes before considering BC admission."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--edges', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = []
    for item in json.loads(args.edges.read_text())['rows']:
        path = Path(item['episode']) / 'frames.jsonl'
        rows = [json.loads(line) for line in path.open()]
        times = np.array([float(r['sim_time']) for r in rows])
        indices = np.array([int(r['command_index']) for r in rows])
        target_errors = [float(np.linalg.norm(np.asarray(r['teacher_target_pose']) -
                           np.asarray(r['executed_target_pose']))) for r in rows]
        images = [p for r in rows for p in r.get('images', {}).values()]
        native = [p for r in rows for p in r.get('native_images', {}).values()]
        # Stored image names are relative to their episode directory.
        missing = [p for p in images + native if not (path.parent / p).is_file()]
        score_success = abs(float(item['official_tier3']) - 75.) < 1e-6
        row = {
            'trial': item['trial'], 'episode': str(path.parent),
            'scene_sha256': item['scene_sha256'], 'frames_sha256': sha256(path),
            'score_tier3': item['official_tier3'], 'score_full_insertion': score_success,
            'frames': len(rows), 'native_frames': sum(bool(r.get('native_images')) for r in rows),
            'sim_duration_s': float(times[-1] - times[0]),
            'time_strictly_increasing': bool(np.all(np.diff(times) > 0)),
            'command_index_nondecreasing': bool(np.all(np.diff(indices) >= 0)),
            'command_index_last': int(indices[-1]),
            'target_max_component_error': float(max(target_errors)),
            'nonzero_perturbation_rows': sum(any(abs(float(x)) > 1e-12 for x in
                r.get('perturbation_xyz_rotvec', [])) for r in rows),
            'student_active_rows': sum(bool(r.get('student_active')) for r in rows),
            'missing_image_files': len(missing),
            'rgb_triplets_complete': all(set(r.get('images', {})) == {'center', 'left', 'right'}
                                          for r in rows),
        }
        row['candidate_pass'] = bool(score_success and row['time_strictly_increasing']
            and row['command_index_nondecreasing'] and row['target_max_component_error'] < 1e-9
            and row['nonzero_perturbation_rows'] == 0 and row['student_active_rows'] == 0
            and row['missing_image_files'] == 0 and row['rgb_triplets_complete'])
        results.append(row)
    report = {
        'schema': 'sc_new_expert_candidate_audit/v1',
        'source_edges': str(args.edges), 'source_edges_sha256': sha256(args.edges),
        'note': 'Candidate integrity only; physical plug TF, setpoint reachability, and causal chunk masks need separate audit before BC admission.',
        'candidate_pass_count': sum(r['candidate_pass'] for r in results),
        'candidate_fail_count': sum(not r['candidate_pass'] for r in results),
        'episodes': results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'pass': report['candidate_pass_count'],
                      'fail': report['candidate_fail_count']}))


if __name__ == '__main__':
    main()
