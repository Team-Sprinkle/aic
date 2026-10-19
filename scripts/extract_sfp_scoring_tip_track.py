#!/usr/bin/env python3
"""Extract scored Gazebo world-to-plug-tip TF for posthoc diagnosis only.

Run inside the official rootless image, which supplies ROS bag readers.
No output from this script is an actor observation or a training label.
"""
import argparse
import json
from pathlib import Path

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def record(edge):
    tr=edge.transform
    return {'xyz_m':[tr.translation.x,tr.translation.y,tr.translation.z],
            'quat_xyzw':[tr.rotation.x,tr.rotation.y,tr.rotation.z,tr.rotation.w]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('bag',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve prior scored tip track')
    reader=rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(a.bag),storage_id='mcap'),
                rosbag2_py.ConverterOptions('cdr','cdr'))
    types={topic.name:get_message(topic.type) for topic in reader.get_all_topics_and_types()}
    rows=[]
    while reader.has_next():
        topic,raw,bag_stamp=reader.read_next()
        if topic!='/scoring/tf':continue
        message=deserialize_message(raw,types[topic])
        found={(edge.header.frame_id,edge.child_frame_id):edge for edge in message.transforms}
        world=found.get(('aic_world','cable_0'))
        tip=found.get(('cable_0','cable_0/sfp_tip_link'))
        if world is None or tip is None:continue
        stamp=tip.header.stamp
        rows.append({'sim_time_s':stamp.sec+stamp.nanosec/1e9,
                     'bag_stamp_ns':bag_stamp,'world_cable':record(world),
                     'cable_tip':record(tip)})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w') as file:
        for row in rows:file.write(json.dumps(row,separators=(',',':'))+'\n')
    print(json.dumps({'rows':len(rows),'first_sim_time_s':rows[0]['sim_time_s'] if rows else None,
                      'last_sim_time_s':rows[-1]['sim_time_s'] if rows else None,
                      'posthoc_scoring_only':True}))


if __name__=='__main__':main()
