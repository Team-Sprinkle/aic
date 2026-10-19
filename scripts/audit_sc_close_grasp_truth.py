#!/usr/bin/env python3
"""Posthoc SC held-plug pose error at a close-grasp rejection; never runtime input."""
import argparse
from bisect import bisect_right
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_local_actor_scored_tip import record_at


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rollout', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((args.rollout / 'scored_stream.json').read_text())
    trace = [json.loads(line) for line in (args.rollout / 'runtime_trace.jsonl').open()]
    rejected = next(row for row in trace if row['event'] == 'close_grasp_update_rejected' and
                    row['reason'] == 'persistent_large_disagreement_reacquisition_required')
    old = next(row for row in trace if row['event'] == 'full_connector_grasp_bound')
    near = rejected['candidate']
    # This event follows the second quiet, fresh-camera consensus window.
    index = trace.index(rejected)
    stamps = [row['sim_time'] for row in trace[:index] if row['event'] == 'close_grasp_observation']
    assert stamps
    seconds = max(stamps)
    edges = data['edges']
    tabletop = 'aic_world|tabletop' if 'aic_world|tabletop' in edges else 'world|tabletop'
    world_base = record_at(edges[tabletop], 0) @ record_at(edges['tabletop|base_link'], 0)
    cable = data['task']['cable_name']
    parent = f'aic_world|{cable}' if f'aic_world|{cable}' in edges else f'world|{cable}'
    tip = f'{cable}|{cable}/sc_tip_link'
    world_tip = record_at(edges[parent], seconds * 1e9) @ record_at(edges[tip], seconds * 1e9)
    states = data['streams']['/aic_controller/controller_state']
    state_times = [row[0] for row in states]
    state_index = bisect_right(state_times, seconds * 1e9) - 1
    assert state_index >= 0
    measured = states[state_index][1]['tcp_pose']
    base_tcp = np.eye(4)
    base_tcp[:3, 3] = [measured['position'][key] for key in 'xyz']
    base_tcp[:3, :3] = Rotation.from_quat([measured['orientation'][key] for key in 'xyzw']).as_matrix()
    actual = np.linalg.inv(world_base @ base_tcp) @ world_tip

    def compare(position, rotation):
        return dict(translation_error_mm=float(np.linalg.norm(np.asarray(position) - actual[:3, 3]) * 1000),
                    orientation_error_deg=float(np.degrees(Rotation.from_matrix(
                        np.asarray(rotation).T @ actual[:3, :3]).magnitude())))

    result = dict(schema='posthoc_sc_close_grasp_truth/v1', rollout=str(args.rollout),
                  observation_time_s=seconds,
                  controller_state_age_ms=float((seconds * 1e9 - state_times[state_index]) / 1e6),
                  original=compare(old['position_tcp_m'], old['rotation_tcp']),
                  close_two_window=compare(near['position_m'], near['rotation']),
                  near_window_translation_span_mm=near['translation_span_mm'],
                  near_window_rotation_span_deg=near['rotation_span_deg'],
                  runtime_privileged_geometry_used=False,
                  note='Simulator geometry is read only after the frozen development rollout; it is not a policy input.')
    assert 0 <= result['controller_state_age_ms'] <= 200
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
