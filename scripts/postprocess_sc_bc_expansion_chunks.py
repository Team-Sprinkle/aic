#!/usr/bin/env python3
"""Admit only fully scored, causal SC teacher episodes from captured groups."""

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path('/var/tmp/chmin_aic_20260918_act')
CANONICAL = ROOT / 'sc_canonical_21_port_tcp_waypoint_chunks.jsonl'
EXISTING = [
    CANONICAL,
    ROOT / 'sc_supplemental_candidates_v2_20260924/port_tcp_waypoint_chunks.jsonl',
    ROOT / 'sc_bc_expansion_batch1_candidates_20260924/physical_waypoint_chunks.jsonl',
    ROOT / 'sc_shared_dagger_secondonly_candidates_20260924/corrective_waypoint_chunks.jsonl',
    ROOT / 'sc_shared_early_dagger_retry_candidates_20260925/corrective_waypoint_chunks.jsonl',
    ROOT / 'sc_diversity_batch1_candidates_20260925/physical_waypoint_chunks.jsonl',
    ROOT / 'sc_diversity_batch1_lift_candidates_20260925/physical_waypoint_chunks.jsonl',
    ROOT / 'sc_diversity_actor_early2mm_candidates_20260925/corrective_waypoint_chunks.jsonl',
    ROOT / 'sc_diversity_batch2_candidates_20260925/physical_waypoint_chunks.jsonl',
    ROOT / 'sc_diversity_batch2_lift_candidates_20260925/physical_waypoint_chunks.jsonl',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--labels-summary', type=Path, required=True)
    parser.add_argument('--additional-existing-labels', type=Path, nargs='*', default=[])
    args = parser.parse_args()
    results = json.loads((args.source / 'chunk_results.json').read_text())
    completed = [Path(row['root']) for row in results if row['complete']]
    trial_names = [name for row in results if row['complete'] for name in row['trials']]
    source_names = set(json.loads((args.source / 'source_manifest.json').read_text())['sources'])
    if len(set(trial_names)) != len(trial_names) or not set(trial_names) <= source_names:
        raise ValueError('Duplicate or unknown trial among complete scored groups')
    missing = sorted(source_names - set(trial_names))
    if missing:
        print(json.dumps({'unscored_or_incomplete_trials': missing}), flush=True)
    if any(not (root / 'edges_joined.json').exists() or
           not (root / 'dynamic_grasp_calibration.json').exists() for root in completed):
        raise ValueError('Scored edges or grasp calibration missing from a complete group')
    commands = [
        ['.pixi/envs/default/bin/python', 'scripts/build_sc_supplemental_candidates.py',
         '--runs', *map(str, completed), '--output-dir', str(args.output_dir)],
        ['.pixi/envs/default/bin/python', 'scripts/build_sc_connector_waypoint_labels.py',
         '--manifest', str(args.output_dir / 'manifest.json'),
         '--port-edges', str(args.output_dir / 'port_edges.json'),
         '--tcp-tip-calibration', 'configs/hierarchical_recovery/sc_tcp_tip_gazebo_tf_251.json',
         '--episode-tip-calibrations', str(args.output_dir / 'physical_grasp_calibrations.json'),
         '--port-opening-calibration', 'configs/hierarchical_recovery/sc_port_opening_gazebo_tf_251.json',
         '--labels-output', str(args.output_dir / 'physical_waypoint_chunks.jsonl'),
         '--summary-output', str(args.labels_summary), '--resample-mode', 'causal_hold'],
        ['.pixi/envs/default/bin/python', 'scripts/admit_sc_bc_candidates.py',
         '--candidate-manifest', str(args.output_dir / 'manifest.json'),
         '--waypoint-labels', str(args.output_dir / 'physical_waypoint_chunks.jsonl'),
         '--verified-manifest', 'outputs/trajectory_datasets/expert_verified/manifest.json',
         '--existing-labels', *map(str, [*EXISTING, *args.additional_existing_labels]),
         '--output', str(args.admission)]
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'capture_coverage.json').write_text(json.dumps({
        'source_trials': len(source_names), 'complete_scored_trials': len(trial_names),
        'incomplete_or_unscored_trials': missing, 'complete_groups': list(map(str, completed)),
    }, indent=2) + '\n')
    (args.output_dir / 'postprocess_commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
