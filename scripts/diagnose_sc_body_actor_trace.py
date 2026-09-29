#!/usr/bin/env python3
"""Posthoc scored SC trace audit; never supplies geometry to the actor."""

import argparse
import bisect
import json
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from audit_sc_port_targets import matrix, pose_matrix
from build_sc_native_pose_labels import chain, project
from inspect_sc_port_tf import extract_edges


def pose(message):
    p, q = message.position, message.orientation
    return [p.x, p.y, p.z, q.x, q.y, q.z, q.w]


def transform(edge):
    t = edge.transform
    return pose_matrix([t.translation.x, t.translation.y, t.translation.z,
                        t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w])


def stamp(message):
    s = message.header.stamp
    return float(s.sec) + 1e-9 * float(s.nanosec)


def nearest(rows, t):
    times = [x['time'] for x in rows]
    k = bisect.bisect_left(times, t)
    return min((rows[i] for i in (max(0, k-1), min(k, len(rows)-1))),
               key=lambda x: abs(x['time']-t))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bag', type=Path, required=True)
    p.add_argument('--port', type=int, choices=(0, 1), required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    edges = extract_edges(a.bag, first_seconds=20, include_camera_chain=True)
    base_world = np.linalg.inv(chain(edges, ['world', 'tabletop', 'base_link']))
    parent = f'task_board/sc_port_{a.port}'
    opening_offset = matrix(json.loads(Path(
        'configs/hierarchical_recovery/sc_port_opening_gazebo_tf_251.json').read_text())['base_to_opening'])
    world_opening = chain(edges, ['aic_world', 'task_board', parent,
                                  parent+'/sc_port_base_link']) @ opening_offset
    base_opening = base_world @ world_opening
    opening_base = np.linalg.inv(base_opening)
    tcp_tip = matrix(json.loads(Path(
        'configs/hierarchical_recovery/sc_tcp_tip_gazebo_tf_251.json').read_text())['tcp_to_sc_tip'])
    mount_tcp = chain(edges, ['tool0', 'cam_mount/cam_mount_link',
                              'ati/base_link', 'ati/tool_link',
                              'gripper/hande_base_link', 'gripper/tcp'])
    tcp_cameras = {}
    for camera in ('center', 'left', 'right'):
        mount_camera = chain(edges, ['tool0', 'cam_mount/cam_mount_link',
                                     f'{camera}_camera/camera_link',
                                     f'{camera}_camera/sensor_link',
                                     f'{camera}_camera/optical'])
        tcp_cameras[camera] = np.linalg.inv(mount_tcp) @ mount_camera
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(a.bag), storage_id='mcap'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    wanted = {'/aic_controller/controller_state', '/aic_controller/pose_commands',
              '/fts_broadcaster/wrench', '/scoring/tf'}
    types = {x.name: get_message(x.type) for x in reader.get_all_topics_and_types()
             if x.name in wanted}
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(wanted)))
    states, commands, tips, forces = [], [], [], []
    while reader.has_next():
        topic, raw, _ = reader.read_next()
        message = deserialize_message(raw, types[topic])
        if topic == '/aic_controller/controller_state':
            states.append({'time': stamp(message), 'tcp': pose(message.tcp_pose),
                           'reference': pose(message.reference_tcp_pose),
                           'tcp_error': list(map(float, message.tcp_error))})
        elif topic == '/aic_controller/pose_commands':
            commands.append({'time': stamp(message), 'tcp': pose(message.pose)})
        elif topic == '/fts_broadcaster/wrench':
            f = message.wrench.force
            forces.append({'time': stamp(message),
                           'norm_n': float(np.linalg.norm([f.x, f.y, f.z]))})
        else:
            parts = {x.child_frame_id: x for x in message.transforms}
            if 'cable_1' in parts and 'cable_1/sc_tip_link' in parts:
                t = parts['cable_1/sc_tip_link']
                tips.append({'time': float(t.header.stamp.sec)+1e-9*t.header.stamp.nanosec,
                             'world_tip': transform(parts['cable_1']) @ transform(t)})
    if not states or not commands or not tips or not forces:
        raise ValueError('Missing scored causal trace topics')
    start = commands[0]['time']
    rows = []
    for second in range(0, 91):
        t = start + second
        if t > min(states[-1]['time'], tips[-1]['time']):
            break
        s, c, tip, f = (nearest(source, t) for source in
                         (states, commands, tips, forces))
        base_tcp = pose_matrix(s['tcp'])
        current_tip = opening_base @ base_world @ tip['world_tip']
        # Commanded connector tip uses the fixed grasp proxy; actual tip
        # remains the scored dynamic TF above.
        proposed_tip = opening_base @ pose_matrix(c['tcp']) @ tcp_tip
        actual_xyz = current_tip[:3, 3] * 1000
        proposed_xyz = proposed_tip[:3, 3] * 1000
        visible = {}
        for camera, tcp_camera in tcp_cameras.items():
            u, v, depth, inside = project(base_opening,
                                           base_tcp @ tcp_camera)
            visible[camera] = {'frustum': inside, 'u_native_px': u,
                               'v_native_px': v, 'depth_m': depth}
        rows.append({'time_from_first_command_s': second,
                     'actual_tip_opening_xyz_mm': actual_xyz.tolist(),
                     'actual_tip_distance_mm': float(np.linalg.norm(actual_xyz)),
                     'commanded_proxy_tip_opening_xyz_mm': proposed_xyz.tolist(),
                     'commanded_proxy_tip_distance_mm': float(np.linalg.norm(proposed_xyz)),
                     'tcp_reference_error_mm': float(np.linalg.norm(
                         np.asarray(s['reference'][:3])-np.asarray(s['tcp'][:3]))*1000),
                     'tcp_state_error_mm': float(np.linalg.norm(s['tcp_error'][:3])*1000),
                     'force_n': f['norm_n'], 'opening_camera_projection': visible,
                     'nearest_stamp_gap_ms': {'state': abs(s['time']-t)*1000,
                                              'command': abs(c['time']-t)*1000,
                                              'tip': abs(tip['time']-t)*1000,
                                              'force': abs(f['time']-t)*1000}})
    report = {'schema': 'sc_body_actor_scored_trace/v1', 'diagnostic_only': True,
              'privileged_geometry_at_autonomous_runtime': False,
              'bag': str(a.bag), 'selected_port': a.port,
              'topic_counts': {'states': len(states), 'commands': len(commands),
                               'tips': len(tips), 'forces': len(forces)},
              'camera_visibility_semantics': 'geometric frustum only; no occlusion ray test or saved RGB',
              'rows_1hz': rows}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'seconds': len(rows), 'topic_counts': report['topic_counts'],
                      'first_distance_mm': rows[0]['actual_tip_distance_mm'],
                      'last_distance_mm': rows[-1]['actual_tip_distance_mm']}))


if __name__ == '__main__':
    main()
