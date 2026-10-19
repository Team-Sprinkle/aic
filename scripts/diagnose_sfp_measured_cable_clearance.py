#!/usr/bin/env python3
"""Posthoc cable/card proximity from measured Gazebo TF; not a contact oracle."""
import argparse
import json
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from tf2_msgs.msg import TFMessage

from audit_sfp_requested_corridor_cable import (
    cable_samples_tcp, nic_solids, signed_clearance, tf_matrix,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--start-s', type=float, default=8.)
    p.add_argument('--end-s', type=float, default=20.)
    p.add_argument('--interval-s', type=float, default=.2)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve previous result')
    bag = next(a.audit.glob('results/bag_trial*/*.mcap'))
    solids = nic_solids(a.audit / 'eval_config.yaml')
    rows = []
    next_t = a.start_s
    with bag.open('rb') as stream:
        for _, _, raw in make_reader(stream).iter_messages(topics=['/scoring/tf']):
            msg = deserialize_message(raw.data, TFMessage)
            if not msg.transforms:
                continue
            stamp = msg.transforms[0].header.stamp
            t = stamp.sec + stamp.nanosec * 1e-9
            if t < next_t:
                continue
            if t > a.end_s:
                break
            edges = {(x.header.frame_id, x.child_frame_id): tf_matrix(x.transform)
                     for x in msg.transforms}
            try:
                # Identity means the returned samples remain in aic_world.
                cable = cable_samples_tcp(edges, np.eye(4))
                proximal = signed_clearance(cable, np.eye(4), solids, 5)
                full = signed_clearance(cable, np.eye(4), solids, 20)
            except (KeyError, RuntimeError):
                continue
            rows.append(dict(sim_s=t,
                             proximal_mm=proximal[0], proximal_link=proximal[1],
                             proximal_card=proximal[2],
                             full_mm=full[0], full_link=full[1],
                             full_card=full[2]))
            next_t += a.interval_s
    result = dict(schema='aic_sfp_measured_cable_clearance_diagnostic/v1',
                  source=str(a.audit), bag=str(bag),
                  posthoc_scoring_only=True, contact_oracle=False,
                  rows=rows)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(samples=len(rows),
                          min_proximal_mm=min((r['proximal_mm'] for r in rows),default=None),
                          min_full_mm=min((r['full_mm'] for r in rows),default=None))))


if __name__ == '__main__':
    main()
