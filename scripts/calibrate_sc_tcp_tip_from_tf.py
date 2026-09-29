#!/usr/bin/env python3
"""Derive the fixed gripper TCP -> SC tip transform from scored Gazebo TF.

The input is the event snapshot from extract_gazebo_sc_joint_seed.py. This
transform is for supervised labels only. It is not an autonomous observation.
"""

import argparse
import json
import math
from pathlib import Path


def qmul(a, b):
    x, y, z, w = a
    X, Y, Z, W = b
    return [w*X+x*W+y*Z-z*Y, w*Y-x*Z+y*W+z*X,
            w*Z+x*Y-y*X+z*W, w*W-x*X-y*Y-z*Z]


def rotate(q, v):
    return qmul(qmul(q, [*v, 0.0]), [-q[0], -q[1], -q[2], q[3]])[:3]


def compose(a, b):
    pa, qa = a
    pb, qb = b
    shift = rotate(qa, pb)
    return [pa[i] + shift[i] for i in range(3)], qmul(qa, qb)


def inverse(a):
    p, q = a
    qi = [-q[0], -q[1], -q[2], q[3]]
    return [-v for v in rotate(qi, p)], qi


def world_pose(tf, name):
    pose = ([0.0]*3, [0.0, 0.0, 0.0, 1.0])
    path = []
    while name in tf:
        if name in path:
            raise ValueError("TF cycle")
        path.append(name)
        edge = tf[name]
        pose = compose((edge["xyz_m"], edge["quat_xyzw"]), pose)
        name = edge["parent"]
    if name != "world":
        raise ValueError(f"TF path ended at {name}, not world")
    return pose


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("snapshot", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    source = json.loads(a.snapshot.read_text())
    tf = source["transforms_at_first_event"]
    tcp = world_pose(tf, "gripper/tcp")
    tip = world_pose(tf, "cable_1/sc_tip_link")
    xyz, quat = compose(inverse(tcp), tip)
    length = math.sqrt(sum(v*v for v in quat))
    quat = [v/length for v in quat]
    result = {
        "schema": "aic_sc_tcp_tip_calibration/v1",
        "source_bag": source["bag"],
        "source_snapshot": str(a.snapshot),
        "event_time_ns": source["events"][0]["time_ns"],
        "method": "inverse(world->gripper/tcp) @ (world->cable_1/sc_tip_link)",
        "tcp_to_sc_tip": {"xyz_m": xyz, "quat_xyzw": quat},
        "training_label_only": True,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(a.output)


if __name__ == "__main__":
    main()
