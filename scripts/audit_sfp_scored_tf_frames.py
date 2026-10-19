#!/usr/bin/env python3
"""List scored TF edges near a requested simulation time; posthoc use only."""
import argparse
import json

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

p = argparse.ArgumentParser()
p.add_argument('bag')
p.add_argument('--time', type=float, default=30)
a = p.parse_args()
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=a.bag, storage_id='mcap'),
       rosbag2_py.ConverterOptions('cdr', 'cdr'))
t = {x.name: get_message(x.type) for x in r.get_all_topics_and_types()}
while r.has_next():
    topic, raw, _ = r.read_next()
    if topic != '/scoring/tf':
        continue
    msg = deserialize_message(raw, t[topic])
    for e in msg.transforms:
        stamp = e.header.stamp.sec + e.header.stamp.nanosec / 1e9
        if abs(stamp - a.time) < .002:
            print(json.dumps({'time': stamp, 'parent': e.header.frame_id,
                              'child': e.child_frame_id,
                              'xyz': [e.transform.translation.x,
                                      e.transform.translation.y,
                                      e.transform.translation.z]}))
