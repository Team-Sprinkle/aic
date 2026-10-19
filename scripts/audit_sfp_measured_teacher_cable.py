#!/usr/bin/env python3
"""Measure the same conservative NIC/cable bound on saved Gazebo teacher runs."""
import argparse
import json
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from tf2_msgs.msg import TFMessage

from audit_sfp_requested_corridor_cable import (cable_samples_tcp, clearance,
                                                 signed_clearance, nic_solids, tf_matrix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plans', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--horizon-s', type=float, default=20)
    parser.add_argument('--signed-distance', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve existing audit')
    cases = []
    measure=signed_clearance if args.signed_distance else clearance
    for n in range(6):
        plan = json.loads((args.plans / f'case{n}.json').read_text())
        audit = Path(plan['audit'])
        bag = next(audit.glob('results/bag_trial*/*.mcap'))
        solids = nic_solids(audit / 'eval_config.yaml')
        rows, last = [], -1e9
        with bag.open('rb') as stream:
            for _, _, raw in make_reader(stream).iter_messages(topics=['/scoring/tf']):
                message = deserialize_message(raw.data, TFMessage)
                if not message.transforms:
                    continue
                header = message.transforms[0].header.stamp
                t = header.sec + header.nanosec * 1e-9
                if t < plan['takeover_sim_s'] - .05 or t > plan['takeover_sim_s'] + args.horizon_s:
                    continue
                if t - last < .1:
                    continue
                last = t
                edges = {(x.header.frame_id, x.child_frame_id): tf_matrix(x.transform)
                         for x in message.transforms}
                try:
                    cable = cable_samples_tcp(edges, np.eye(4))
                except RuntimeError:
                    continue
                proximal = measure(cable, np.eye(4), solids, 5)
                full = measure(cable, np.eye(4), solids, 20)
                rows.append({'time_s': t, 'proximal_mm': proximal[0], 'full_mm': full[0],
                             'proximal_link': proximal[1], 'full_link': full[1]})
        if not rows:
            raise RuntimeError(f'No scored cable TF for case {n}')
        cases.append({'case': n, 'scene_group': plan['scene_group'], 'bag': str(bag),
                      'samples': len(rows),
                      'min_proximal_mm': min(x['proximal_mm'] for x in rows),
                      'min_full_mm': min(x['full_mm'] for x in rows),
                      'start': rows[0], 'trace': rows})
        print(n, len(rows), cases[-1]['min_full_mm'], flush=True)
    args.output.write_text(json.dumps({
        'schema': 'aic_sfp_measured_teacher_cable_clearance/v1',
        'signed_distance': args.signed_distance,
        'horizon_s': args.horizon_s,
        'note': 'Conservative lower bound is not actual contact force or penetration.',
        'cases': cases}, indent=2) + '\n')


if __name__ == '__main__':
    main()
