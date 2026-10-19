#!/usr/bin/env python3
"""Score frozen SC transport-head proposals against posthoc waiting-pose truth.

The saved head probabilities are causal. Teacher geometry is opened only for
offline labels and never used to choose a runtime proposal. This is diagnosis
on development traces, not a promoted threshold or a final-scene evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def route_error_mm(xyz: np.ndarray, path: np.ndarray) -> np.ndarray:
    distances = []
    for start, end in zip(path[:-1], path[1:]):
        step = end - start
        length2 = float(step @ step)
        if length2 < 1e-14:
            continue
        alpha = np.clip(((xyz - start) @ step) / length2, 0, 1)
        distances.append(np.linalg.norm(xyz - (start + alpha[:, None] * step), axis=1) * 1000)
    return np.min(np.stack(distances), axis=0)


def load_episode(record: dict) -> dict:
    audit = Path(record['audit'])
    trace = [json.loads(line) for line in (audit / 'runtime_trace.jsonl').open()]
    proposals = [row for row in trace if row['event'] == 'proposal']
    head = record['transport_head_trace']
    if len(proposals) != len(head):
        raise ValueError(f'Proposal/head length mismatch: {audit}')
    if not np.allclose([r['sim_time'] for r in proposals],
                       [r['sim_time'] for r in head], atol=1e-3):
        raise ValueError(f'Proposal/head timestamp mismatch: {audit}')
    waypoints = json.loads(Path(record['posthoc_goal_plan']).read_text())['waypoints']
    path = np.asarray([r['tcp_pose']['position'] for r in waypoints], dtype=float)
    goal = waypoints[-1]['tcp_pose']
    goal_xyz = np.asarray(goal['position'], dtype=float)
    goal_rot = Rotation.from_quat(goal['orientation_xyzw'])
    state = np.asarray([r['state'] for r in proposals], dtype=float)
    xyz = state[:, :3]
    position_mm = np.linalg.norm(xyz - goal_xyz, axis=1) * 1000
    angle_deg = np.asarray([
        (Rotation.from_quat(pose[3:7]).inv() * goal_rot).magnitude() * 180 / np.pi
        for pose in state
    ])
    route_mm = route_error_mm(xyz, path)
    max_route_mm = np.maximum.accumulate(route_mm)
    probability = np.asarray([r['transport_probability'] for r in head], dtype=float)
    time = np.asarray([r['sim_time'] for r in head], dtype=float)
    return dict(episode=record['episode'], audit=str(audit), time=time,
                probability=probability, position_mm=position_mm,
                angle_deg=angle_deg, route_mm=route_mm,
                max_route_mm=max_route_mm)


def first_stop(episode: dict, threshold: float, streak_required: int,
               min_time_s: float) -> int | None:
    streak = 0
    for index, (time, probability) in enumerate(zip(episode['time'], episode['probability'])):
        streak = streak + 1 if time >= min_time_s and probability >= threshold else 0
        if streak >= streak_required:
            return index
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--min-time-s', type=float, default=15.)
    parser.add_argument('--streak', type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve previous result; choose a new output path')
    frozen = json.loads(args.comparison.read_text())
    if frozen['schema'] != 'aic_frozen_sc_stop_comparison/v1' or frozen['reserved_final_opened']:
        raise ValueError('Expected sealed seven-path development comparison')
    episodes = [load_episode(record) for record in frozen['episodes']]
    # Posthoc labels define an invitation to inspect, not permission to insert.
    # 80 mm follows the existing SC visual pre-gate. Orientation/route limits
    # are deliberately stated here; both must be checked on new scenes later.
    for episode in episodes:
        episode['coarse_safe'] = ((episode['position_mm'] <= 80) &
                                  (episode['angle_deg'] <= 10) &
                                  (episode['max_route_mm'] <= 50))
        episode['strict_safe'] = ((episode['position_mm'] <= 10) &
                                  (episode['angle_deg'] <= 3) &
                                  (episode['max_route_mm'] <= 50))
    thresholds = [.1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .975, .99]
    sweeps = []
    for threshold in thresholds:
        rows = []
        for episode in episodes:
            index = first_stop(episode, threshold, args.streak, args.min_time_s)
            first_safe = np.flatnonzero(episode['coarse_safe'])
            first_strict = np.flatnonzero(episode['strict_safe'])
            row = dict(episode=episode['episode'], audit=episode['audit'],
                       first_coarse_safe_s=float(episode['time'][first_safe[0]]) if len(first_safe) else None,
                       first_strict_safe_s=float(episode['time'][first_strict[0]]) if len(first_strict) else None,
                       proposed_s=None if index is None else float(episode['time'][index]))
            if index is not None:
                row.update(position_mm=float(episode['position_mm'][index]),
                           angle_deg=float(episode['angle_deg'][index]),
                           max_route_mm=float(episode['max_route_mm'][index]),
                           coarse_safe=bool(episode['coarse_safe'][index]),
                           strict_safe=bool(episode['strict_safe'][index]))
            rows.append(row)
        sweeps.append(dict(threshold=threshold,
                           proposed=sum(row['proposed_s'] is not None for row in rows),
                           safe_proposals=sum(row.get('coarse_safe', False) for row in rows),
                           false_proposals=sum(row['proposed_s'] is not None and
                                               not row.get('coarse_safe', False) for row in rows),
                           missed_safe=sum(row['first_coarse_safe_s'] is not None and
                                           row['proposed_s'] is None for row in rows),
                           rows=rows))
    summary = dict(schema='aic_frozen_transport_arrival_diagnostic/v1',
                   source=str(args.comparison), head_sha256=frozen['transport_head_sha256'],
                   episodes=len(episodes), episode_groups=len({row['episode'] for row in frozen['episodes']}),
                   labels='posthoc teacher standoff and route; never actor inputs',
                   proposal_label=dict(position_mm=80, orientation_deg=10,
                                       max_route_deviation_mm=50),
                   min_time_s=args.min_time_s, consecutive_decisions=args.streak,
                   thresholds=sweeps, reserved_final_opened=False,
                   threshold_selection_validated_on_new_scenes=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    for row in sweeps:
        print(f"p≥{row['threshold']:.3f}: safe={row['safe_proposals']}, "
              f"false={row['false_proposals']}, missed_safe={row['missed_safe']}")


if __name__ == '__main__':
    main()
