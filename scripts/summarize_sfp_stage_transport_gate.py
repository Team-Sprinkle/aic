#!/usr/bin/env python3
"""Summarize stage-label support, measured cable contacts, and replay mismatch."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def load(path):
    return json.loads(path.read_text())


def contact_peak(audit, start, end):
    peak = {'force_n': 0., 'time_s': None, 'pair': None}
    count = 0
    for line in (audit / 'corrected_contact_timeline.jsonl').open():
        row = json.loads(line)
        time = row['sim_ns'] * 1e-9
        if time < start or time > end:
            continue
        for contact in row['contacts']:
            pair = contact['pair']
            if 'cable_0::link_' not in pair or 'nic_card_mount' not in pair:
                continue
            count += 1
            force = contact['max_point_normal_force_n']
            if force > peak['force_n']:
                peak = {'force_n': force, 'time_s': time, 'pair': pair}
    peak['sampled_events'] = count
    return peak


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-audit', type=Path, required=True)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--new', type=Path, action='append', required=True)
    p.add_argument('--route-replays', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve earlier decision')
    old = load(a.old_audit)['records']
    pilot = load(a.pilot)['records']
    fresh = [load(path)['records'][0] for path in a.new]
    train = [r for r in old if r['split'] == 'train']
    development = [r for r in old if r['split'] == 'validation']
    extra = [r for r in pilot if r['official_full_insertion']] + fresh
    alltrain = train + extra
    scenes = [r['scene'] for r in alltrain]
    contact = []
    for row in fresh:
        audit = Path(row['source_scored_stream']).parent
        take = load(audit / 'corrective_teacher_takeover.json')['takeover_sim_time_s']
        onset = take + row['onset_after_takeover_s']
        contact.append({
            'scene': row['scene'], 'episode_id': row['episode_id'],
            'takeover_s': take, 'onset_s': onset,
            'label_end_s': onset + row['route_duration_s'],
            'short_label_peak': contact_peak(audit, onset, onset + row['route_duration_s']),
            'first_five_seconds_peak': contact_peak(audit, take, take + 5),
            'full_teacher_peak': contact_peak(audit, take, 1e10),
        })
    names = ['side_down_1', 'side_down_2', 'side_sweep_1',
             'side_sweep_2', 'side_sweep_3']
    starts = []
    for name in names:
        folder = a.route_replays / name
        row = load(folder / 'moveit_plan_request.json')
        starts.append({'name': name, 'tcp_base_m': row['tcp_base_m'],
                       'quat_xyzw': row['tcp_quat_xyzw'],
                       'config_sha256': hashlib.sha256((folder / 'eval_config.yaml').read_bytes()).hexdigest(),
                       'route_passed': load(folder / 'cable_side_gate.json')['pass_all']})
    pairs = []
    for x, y in itertools.combinations(starts, 2):
        qx = Rotation.from_quat(x['quat_xyzw'])
        qy = Rotation.from_quat(y['quat_xyzw'])
        pairs.append({'left': x['name'], 'right': y['name'],
                      'tcp_distance_mm': float(np.linalg.norm(np.array(x['tcp_base_m']) -
                                                            np.array(y['tcp_base_m'])) * 1000),
                      'orientation_distance_deg': float(np.rad2deg((qx.inv() * qy).magnitude()))})
    output = {
        'schema': 'aic_sfp_stage_transport_gate_decision/v1',
        'train_distinct_reset_count': len(set(scenes)),
        'train_usable_count': sum(r['usable'] for r in alltrain),
        'development_distinct_reset_count': len(set(r['scene'] for r in development)),
        'development_usable_count': sum(r['usable'] for r in development),
        'development_independent_direction_pass': sum(
            r['initial_vs_later_20mm_cosine'] is not None and
            r['initial_vs_later_20mm_cosine'] >= .7 for r in development),
        'old_boundary_to_onset_above_5mm_count': sum(
            r['boundary_to_onset_mm'] > 5 for r in old),
        'all_official_full_insertions': all(r['official_full_insertion'] for r in alltrain + development),
        'all_official_contact_clear': all(r.get('official_contact_score') == 0
                                          for r in alltrain + development),
        'fresh_contact': contact,
        'route_replay_starts': starts,
        'route_replay_pairwise': pairs,
        'stage_label_gate_passed': False,
        'reason': 'Full-continuation cable force exceeds 10 N in a successful new teacher run; '
                  'onset labels are not causal actions at all takeover states.',
        'matched_route_ranking_gate_passed': False,
        'route_ranking_reason': 'Same-config starts are 11.8-31.1 mm apart in TCP position; '
                                'no matched-state route outcome comparison exists.',
        'actor_training_allowed': False,
        'final_scenes_opened': False,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({k: v for k, v in output.items() if k not in (
        'fresh_contact', 'route_replay_starts', 'route_replay_pairwise')}))


if __name__ == '__main__':
    main()
