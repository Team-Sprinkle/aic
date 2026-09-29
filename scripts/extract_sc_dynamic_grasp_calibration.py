#!/usr/bin/env python3
"""Sample scored physical SC tip TF to audit episode-specific grasp geometry.

These scored transforms are supervision/diagnostics only. They must never be
used by an autonomous actor or to select its image crops.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from scipy.spatial.transform import Rotation
from tf2_msgs.msg import TFMessage

from audit_sc_port_targets import pose_matrix, pose_list
from build_sc_native_pose_labels import chain


def edge_matrix(edge):
    t = edge.transform
    return pose_matrix([t.translation.x, t.translation.y, t.translation.z,
                        t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w])


def scored_tip_near(reader, row):
    target = float(row['sim_time'])
    # The collector's wall timestamp trails the scored TF record by about
    # 0.1 s in the checked bag. Seek before it and verify the *simulation*
    # stamp, never assume wall and simulation clocks match exactly.
    for lookback in (.5, 2.):
        reader.seek(int((float(row['wall_time']) - lookback) * 1e9))
        previous = None
        for _ in range(20000):
            if not reader.has_next():
                break
            _, raw, _ = reader.read_next()
            message = deserialize_message(raw, TFMessage)
            edges = {x.child_frame_id: x for x in message.transforms}
            if 'cable_1' not in edges or 'cable_1/sc_tip_link' not in edges:
                continue
            stamp = edges['cable_1/sc_tip_link'].header.stamp
            sim = stamp.sec + 1e-9 * stamp.nanosec
            candidate = (sim, edge_matrix(edges['cable_1']) @ edge_matrix(edges['cable_1/sc_tip_link']))
            if sim >= target:
                choices = [candidate] + ([previous] if previous is not None else [])
                selected = min(choices, key=lambda x: abs(x[0] - target))
                if abs(selected[0] - target) <= .01:
                    return selected
                break
            previous = candidate
    raise ValueError(f"No scored TF within 10 ms at simulation time {target}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--edges', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--trials', nargs='*')
    args = p.parse_args()
    chosen = set(args.trials or [])
    episodes = []
    for item in json.loads(args.edges.read_text())['rows']:
        if chosen and item['trial'] not in chosen:
            continue
        frames = [json.loads(x) for x in (Path(item['episode']) / 'frames.jsonl').open()]
        native = [r for r in frames if r.get('native_images')]
        picks = [frames[0], native[len(native) // 2] if native else frames[len(frames) // 2], frames[-1]]
        world_base = chain(item['edges'], ['world', 'tabletop', 'base_link'])
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=item['bag'], storage_id='mcap'),
                    rosbag2_py.ConverterOptions('cdr', 'cdr'))
        reader.set_filter(rosbag2_py.StorageFilter(topics=['/scoring/tf']))
        samples = []
        for row in picks:
            time, world_tip = scored_tip_near(reader, row)
            base_tip = np.linalg.inv(world_base) @ world_tip
            tcp_tip = np.linalg.inv(pose_matrix(row['state'][:7])) @ base_tip
            samples.append({'frame': row['frame'], 'sim_time': row['sim_time'],
                            'tf_time_delta_ms': abs(time - row['sim_time']) * 1000,
                            'tcp_to_physical_sc_tip': pose_list(tcp_tip)})
        first = pose_matrix(samples[0]['tcp_to_physical_sc_tip'])
        for sample in samples:
            current = pose_matrix(sample['tcp_to_physical_sc_tip'])
            sample['translation_change_from_first_mm'] = float(np.linalg.norm(
                current[:3, 3] - first[:3, 3]) * 1000)
            sample['orientation_change_from_first_deg'] = float((
                Rotation.from_matrix(first[:3, :3]).inv() *
                Rotation.from_matrix(current[:3, :3])).magnitude() * 180 / np.pi)
        episodes.append({'trial': item['trial'], 'tier3': item['official_tier3'],
                         'scene_sha256': item['scene_sha256'], 'samples': samples})
        print(json.dumps({'trial': item['trial'], 'tip_drift_mm':
            round(max(s['translation_change_from_first_mm'] for s in samples), 4)}), flush=True)
    report = {'schema': 'sc_dynamic_grasp_calibration/v1', 'training_only': True,
              'source_edges': str(args.edges), 'episodes': episodes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
