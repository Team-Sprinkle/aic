#!/usr/bin/env python3
"""Time RGB rail-line yaw refinement after frozen visual model predictions."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit_sc_port_targets import pose_matrix
from build_sc_native_pose_labels import chain
from evaluate_sc_native_triangulation import triangulate
from evaluate_sc_orientation_landmarks import POINTS, orientation
from evaluate_sc_board_lines_yaw import CAMERAS, camera_yaw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--orientation-predictions', type=Path, required=True)
    p.add_argument('--position-predictions', type=Path, required=True)
    p.add_argument('--training-calibration-edges', type=Path, required=True)
    p.add_argument('--board-port-calibration-json', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=100)
    args = p.parse_args()
    rows = [json.loads(x) for x in args.labels.open()]
    orientation_rows = [json.loads(x) for x in args.orientation_predictions.open()]
    position_rows = [json.loads(x) for x in args.position_predictions.open()]
    if ([(r['trial'], r['frame']) for r in rows] !=
            [(r['trial'], r['frame']) for r in orientation_rows] or
            [(r['trial'], r['frame']) for r in rows] !=
            [(r['trial'], r['frame']) for r in position_rows]):
        raise ValueError('Identity mismatch')
    source = json.loads(args.training_calibration_edges.read_text())['rows'][0]['edges']
    tool_tcp = chain(source, ['tool0', 'cam_mount/cam_mount_link', 'ati/base_link',
                              'ati/tool_link', 'gripper/hande_base_link', 'gripper/tcp'])
    offsets = {}
    for cam in CAMERAS:
        tool_optical = chain(source, ['tool0', 'cam_mount/cam_mount_link',
            f'{cam}_camera/camera_link', f'{cam}_camera/sensor_link', f'{cam}_camera/optical'])
        offsets[cam] = np.linalg.inv(tool_tcp) @ tool_optical
    mount = Rotation.from_quat(json.loads(args.board_port_calibration_json.read_text())['board_to_port_xyzw'])
    indices = np.linspace(0, len(rows) - 1, min(args.samples, len(rows))).astype(int)
    timings = []
    for index in indices:
        row, pred, position = rows[index], orientation_rows[index], position_rows[index]
        cameras = {cam: pose_matrix(row['state'][:7]) @ offset for cam, offset in offsets.items()}
        start = time.perf_counter()
        points = {name: triangulate({cam: pred['cameras'][cam][name] for cam in CAMERAS}, cameras)
                  for name in POINTS}
        prior = (orientation(points) * mount.inv()).as_euler('xyz')[2]
        for cam in CAMERAS:
            camera_yaw(row['images'][cam], position['cameras'][cam]['opening_xy'], cameras[cam], prior)
        timings.append((time.perf_counter() - start) * 1000)
    measured = timings[10:] if len(timings) > 20 else timings
    report = {'schema': 'sc_board_lines_offline_latency/v1',
              'scope': 'three native JPEG decodes, Canny/Hough, RGB landmark ray solve, yaw refinement; excludes upstream CNNs, ROS, actor and command conversion',
              'samples': len(timings), 'warmup_excluded': len(timings) - len(measured),
              'p50_ms': float(np.percentile(measured, 50)),
              'p95_ms': float(np.percentile(measured, 95)),
              'p99_ms': float(np.percentile(measured, 99))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
