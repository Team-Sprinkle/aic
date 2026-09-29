#!/usr/bin/env python3
"""Extract initial scored SFP/NIC/board TF for a Gazebo target-frame audit."""

import argparse
import json
from pathlib import Path

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def extract(path, first_seconds=3):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id='mcap'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {topic.name: get_message(topic.type) for topic in reader.get_all_topics_and_types()}
    found = {}; first = None
    while reader.has_next():
        topic, raw, stamp = reader.read_next()
        if first is None:
            first = stamp
        if stamp > first + first_seconds * 1e9:
            break
        if topic not in {'/tf', '/tf_static', '/scoring/tf'}:
            continue
        for edge in deserialize_message(raw, types[topic]).transforms:
            parent, child = edge.header.frame_id, edge.child_frame_id
            if not any(s in parent + ' ' + child for s in
                       ('task_board', 'nic_card', 'sfp_port', 'sfp_tip', 'cable_',
                        'gripper/tcp', 'base_link', 'tabletop', 'aic_world')):
                continue
            key = (topic, parent, child)
            if key not in found:
                tr = edge.transform
                found[key] = {'topic': topic, 'parent': parent, 'child': child,
                              'xyz_m': [tr.translation.x, tr.translation.y, tr.translation.z],
                              'quat_xyzw': [tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w]}
    return list(found.values())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('bag', type=Path)
    p.add_argument('--first-seconds', type=float, default=3)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    edges = extract(a.bag, a.first_seconds)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({'bag': str(a.bag), 'edges': edges}, indent=2) + '\n')
    print(json.dumps({'edges': len(edges), 'sfp_edges': sum('sfp_port' in x['child'] for x in edges)}))
