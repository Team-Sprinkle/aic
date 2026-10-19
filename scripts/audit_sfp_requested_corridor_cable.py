#!/usr/bin/env python3
"""Audit a planned SFP corridor with takeover cable geometry carried rigidly by TCP.

This is a conservative screening proxy, not a cable simulation.  It preserves
the exact measured cable shape at takeover and moves it with each planned TCP
pose.  Flexible cable motion may differ substantially in Gazebo.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from scipy.spatial.transform import Rotation
from tf2_msgs.msg import TFMessage

from aic_teacher_official.expert_generator.collision_scene import object_geometries_from_gazebo_assets
from audit_sc_port_targets import matrix as calibration_matrix


def tf_matrix(tf):
    m = np.eye(4)
    p, q = tf.translation, tf.rotation
    m[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    m[:3, 3] = [p.x, p.y, p.z]
    return m


def pose_matrix(xyz, quat):
    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat(quat).as_matrix()
    m[:3, 3] = xyz
    return m


def take_tf(bag, target_s):
    best = None
    with bag.open('rb') as stream:
        for _, _, raw in make_reader(stream).iter_messages(topics=['/scoring/tf']):
            msg = deserialize_message(raw.data, TFMessage)
            if not msg.transforms:
                continue
            first = msg.transforms[0].header.stamp
            t = first.sec + first.nanosec * 1e-9
            error = abs(t - target_s)
            if best is None or error < best[0]:
                edges = {(x.header.frame_id, x.child_frame_id): tf_matrix(x.transform)
                         for x in msg.transforms}
                best = (error, t, edges)
            if t > target_s + 0.1:
                break
    if best is None or best[0] > .03:
        raise RuntimeError(f'No cable TF within 30 ms of takeover: {best}')
    return best


def cable_samples_tcp(edges, tcp_world):
    roots = [key for key in edges if key[0] == 'aic_world' and key[1].startswith('cable_')]
    roots = [key for key in roots if (key[1], f'{key[1]}/link_1') in edges]
    if len(roots) != 1:
        raise RuntimeError(f'Expected one scoring cable root, found {roots}')
    root = roots[0]
    inv = np.linalg.inv(tcp_world)
    samples = []
    for i in range(1, 21):
        link_key = (root[1], f'{root[1]}/link_{i}')
        if link_key not in edges:
            raise RuntimeError(f'Missing {link_key}')
        link = edges[root] @ edges[link_key]
        length = .036 if i == 1 else .048
        offsets = np.linspace(-length / 2, length / 2, 9) + (.006 if i == 1 else 0)
        xyz = link[:3, 3][None, :] + offsets[:, None] * link[:3, 2][None, :]
        tcp_xyz = (inv[:3, :3] @ xyz.T).T + inv[:3, 3]
        for point in tcp_xyz:
            samples.append((i, point, length / 16 + .002))
    return samples


def nic_solids(config):
    root = Path(__file__).resolve().parents[1]
    objects = [o for o in object_geometries_from_gazebo_assets(config, repo_root=root)
               if o.role == 'nic_card']
    if not objects:
        raise RuntimeError('No NIC collision objects')
    centers = np.asarray([o.pose.position for o in objects], float)
    inv_rot = np.asarray([Rotation.from_quat(o.pose.orientation_xyzw).as_matrix().T
                          for o in objects])
    half = np.asarray([o.dimensions if o.shape == 'box'
                       else [2 * o.dimensions[1], 2 * o.dimensions[1], o.dimensions[0]]
                       for o in objects], float) / 2
    return objects, centers, inv_rot, half


def clearance(sample_tcp, world_tcp, solids, max_link):
    objects, centers, inv_rot, half = solids
    selected = [(idx, p, margin) for idx, p, margin in sample_tcp if idx <= max_link]
    xyz = np.asarray([x[1] for x in selected])
    world = (world_tcp[:3, :3] @ xyz.T).T + world_tcp[:3, 3]
    local = np.einsum('nij,pnj->pni', inv_rot, world[:, None, :] - centers[None, :, :])
    gaps = np.linalg.norm(np.maximum(np.abs(local) - half[None, :, :], 0), axis=2)
    gaps -= np.asarray([x[2] for x in selected])[:, None]
    p, o = np.unravel_index(np.argmin(gaps), gaps.shape)
    return float(gaps[p, o] * 1000), selected[p][0], objects[o].name


def signed_clearance(sample_tcp, world_tcp, solids, max_link):
    """Box signed distance with cable radius; avoids zero-depth saturation."""
    objects, centers, inv_rot, half = solids
    selected = [(idx, p, margin) for idx, p, margin in sample_tcp if idx <= max_link]
    xyz = np.asarray([x[1] for x in selected])
    world = (world_tcp[:3, :3] @ xyz.T).T + world_tcp[:3, 3]
    local = np.einsum('nij,pnj->pni', inv_rot, world[:, None, :] - centers[None, :, :])
    q = np.abs(local) - half[None, :, :]
    distance = (np.linalg.norm(np.maximum(q, 0), axis=2) +
                np.minimum(np.max(q, axis=2), 0))
    distance -= np.asarray([x[2] for x in selected])[:, None]
    p, o = np.unravel_index(np.argmin(distance), distance.shape)
    return float(distance[p, o] * 1000), selected[p][0], objects[o].name


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plans', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--signed-distance', action='store_true',
                   help='Measure inside-box depth instead of saturating inside at minus cable radius')
    args = p.parse_args()
    if args.output.exists():
        p.error('Preserve existing audit')
    root = Path(__file__).resolve().parents[1]
    cal = json.loads((root / 'configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json').read_text())
    world_base = calibration_matrix(cal['world_to_base'])
    cases = []
    measure = signed_clearance if args.signed_distance else clearance
    for n in range(6):
        plan = json.loads((args.plans / f'case{n}.json').read_text())
        audit = Path(plan['audit'])
        bag = next(audit.glob('results/bag_trial*/*.mcap'))
        error, stamp, edges = take_tf(bag, plan['takeover_sim_s'])
        start_tcp_world = world_base @ pose_matrix(plan['start_tcp_base_m'], plan['start_tcp_quat_xyzw'])
        cable = cable_samples_tcp(edges, start_tcp_world)
        solids = nic_solids(audit / 'eval_config.yaml')
        variants = {}
        for name, route in plan['route_variants'].items():
            if not route['success']:
                variants[name] = {'moveit_success': False}
                continue
            rows = []
            for s in route['samples']:
                tcp_world = world_base @ pose_matrix(s['tcp_base_m'], s['tcp_quat_xyzw'])
                proximal = measure(cable, tcp_world, solids, 5)
                full = measure(cable, tcp_world, solids, 20)
                rows.append({'proximal_mm': proximal[0], 'proximal_link': proximal[1],
                             'proximal_object': proximal[2], 'full_mm': full[0],
                             'full_link': full[1], 'full_object': full[2]})
            min_prox = min(enumerate(rows), key=lambda x: x[1]['proximal_mm'])
            min_full = min(enumerate(rows), key=lambda x: x[1]['full_mm'])
            variants[name] = {'moveit_success': True, 'sample_count': len(rows),
                              'start_proximal_mm': rows[0]['proximal_mm'],
                              'minimum_proximal_mm': min_prox[1]['proximal_mm'],
                              'minimum_proximal_sample': min_prox[0],
                              'minimum_proximal_link': min_prox[1]['proximal_link'],
                              'minimum_proximal_object': min_prox[1]['proximal_object'],
                              'start_full_mm': rows[0]['full_mm'],
                              'minimum_full_mm': min_full[1]['full_mm'],
                              'minimum_full_sample': min_full[0],
                              'minimum_full_link': min_full[1]['full_link'],
                              'minimum_full_object': min_full[1]['full_object'],
                              'trace': rows}
        cases.append({'case': n, 'scene_group': plan['scene_group'],
                      'request_card': plan['request_card'], 'request_port': plan['request_port'],
                      'bag': str(bag), 'takeover_tf_s': stamp,
                      'takeover_tf_error_ms': error * 1000,
                      'nic_collision_shapes': len(solids[0]), 'variants': variants})
    result = {'schema': 'aic_sfp_requested_corridor_rigid_cable_proxy/v1',
              'method': 'measured takeover cable shape held rigid to planned TCP; NIC collision solids represented as oriented boxes',
              'signed_distance': args.signed_distance,
              'limit': 'Flexible cable deformation, tension, dynamic snag, and other obstacles are not modeled. This is an offline rejection screen, not a safe-route certificate.',
              'cases': cases, 'final_scenes_opened': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for row in cases:
        print(row['case'], {name: (round(v['minimum_proximal_mm'], 1),
                                   round(v['minimum_full_mm'], 1))
                            for name, v in row['variants'].items() if v['moveit_success']})


if __name__ == '__main__':
    main()
