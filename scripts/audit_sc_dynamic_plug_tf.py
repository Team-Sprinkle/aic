#!/usr/bin/env python3
"""Compare time-varying scored SC plug TF with the fixed TCP-to-tip proxy.

This is a diagnostic of label fidelity. Scored TF must never enter actor
observations or autonomous crop selection.
"""

import argparse
import bisect
import json
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from scipy.spatial.transform import Rotation
from tf2_msgs.msg import TFMessage

from audit_sc_port_targets import matrix, pose_matrix
from build_sc_native_pose_labels import chain


def transform(edge):
    tr = edge.transform
    return pose_matrix([tr.translation.x, tr.translation.y, tr.translation.z,
                        tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w])


def stats(values):
    return {'count': len(values), 'median': float(np.median(values)) if values else None,
            'p95': float(np.percentile(values, 95)) if values else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--edges', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--trials', nargs='*')
    args = p.parse_args()
    chosen = set(args.trials or [])
    tcp_tip = matrix(json.loads(Path('configs/hierarchical_recovery/sc_tcp_tip_gazebo_tf_251.json').read_text())['tcp_to_sc_tip'])
    opening_offset = matrix(json.loads(Path('configs/hierarchical_recovery/sc_port_opening_gazebo_tf_251.json').read_text())['base_to_opening'])
    episodes = []
    for item in json.loads(args.edges.read_text())['rows']:
        if chosen and item['trial'] not in chosen:
            continue
        e = item['edges']
        parent = f"task_board/{item['target_module_name']}"
        world_port = chain(e, ['aic_world', 'task_board', parent, parent + '/sc_port_base_link']) @ opening_offset
        world_base = chain(e, ['world', 'tabletop', 'base_link'])
        frames = [json.loads(line) for line in (Path(item['episode']) / 'frames.jsonl').open()]
        selected = [r for r in frames if r.get('native_images')] + [frames[-1]]
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=item['bag'], storage_id='mcap'),
                    rosbag2_py.ConverterOptions('cdr', 'cdr'))
        reader.set_filter(rosbag2_py.StorageFilter(topics=['/scoring/tf']))
        stamps, dynamic = [], []
        while reader.has_next():
            _, raw, _ = reader.read_next()
            message = deserialize_message(raw, TFMessage)
            parts = {edge.child_frame_id: edge for edge in message.transforms}
            if 'cable_1' not in parts or 'cable_1/sc_tip_link' not in parts:
                continue
            tip_edge = parts['cable_1/sc_tip_link']
            stamp = tip_edge.header.stamp.sec + 1e-9 * tip_edge.header.stamp.nanosec
            stamps.append(stamp)
            dynamic.append(transform(parts['cable_1']) @ transform(tip_edge))
        if not stamps:
            raise ValueError(f"No scored dynamic SC tip TF: {item['trial']}")
        measurements = []
        for row in selected:
            stamp = float(row['sim_time'])
            index = bisect.bisect_left(stamps, stamp)
            index = min([max(index - 1, 0), min(index, len(stamps) - 1)],
                        key=lambda k: abs(stamps[k] - stamp))
            actual = dynamic[index]
            proxy = world_base @ pose_matrix(row['state'][:7]) @ tcp_tip
            actual_port = np.linalg.inv(world_port) @ actual
            proxy_port = np.linalg.inv(world_port) @ proxy
            delta = np.linalg.inv(world_port)[:3, :3] @ (proxy[:3, 3] - actual[:3, 3]) * 1000
            angle = (Rotation.from_matrix(proxy[:3, :3]).inv() *
                     Rotation.from_matrix(actual[:3, :3])).magnitude() * 180 / np.pi
            measurements.append({'frame': row['frame'], 'sim_time': stamp,
                'tf_time_delta_ms': abs(stamps[index] - stamp) * 1000,
                'dynamic_axial_mm': float(actual_port[2, 3] * 1000),
                'dynamic_lateral_mm': float(np.linalg.norm(actual_port[:2, 3]) * 1000),
                'proxy_axial_mm': float(proxy_port[2, 3] * 1000),
                'proxy_lateral_mm': float(np.linalg.norm(proxy_port[:2, 3]) * 1000),
                'proxy_dynamic_translation_mm': float(np.linalg.norm(delta)),
                'proxy_dynamic_orientation_deg': float(angle)})
        near = [r for r in measurements if abs(r['dynamic_axial_mm']) <= 30]
        episodes.append({'trial': item['trial'], 'tier3': item['official_tier3'],
            'scored_tf_samples': len(stamps), 'native_comparisons': len(measurements),
            'tf_time_delta_ms': stats([r['tf_time_delta_ms'] for r in measurements]),
            'all_proxy_dynamic_translation_mm': stats([r['proxy_dynamic_translation_mm'] for r in measurements]),
            'near_proxy_dynamic_translation_mm': stats([r['proxy_dynamic_translation_mm'] for r in near]),
            'near_proxy_dynamic_orientation_deg': stats([r['proxy_dynamic_orientation_deg'] for r in near]),
            'last': measurements[-1], 'measurements': measurements})
    report = {'schema': 'sc_dynamic_tip_vs_fixed_tcp_proxy/v1',
              'diagnostic_only': True, 'scored_tf_runtime_actor_input': False,
              'episodes': episodes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'episodes': len(episodes), 'last_proxy_dynamic_mm':
        {r['trial']: round(r['last']['proxy_dynamic_translation_mm'], 3) for r in episodes}}))


if __name__ == '__main__':
    main()
