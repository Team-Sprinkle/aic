#!/usr/bin/env python3
"""Diagnostic SC board-yaw regression from RGB blue-port silhouettes.

Scored TF supplies training/evaluation labels only. Features come from the
observation-selected native crop, measured robot state, and task encoding.
This is a grouped development ablation, not a deployable actor input.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from audit_sc_port_targets import matrix, pose_matrix


CAMERAS = ('center', 'left', 'right')


def shape_features(path):
    with Image.open(path) as image:
        rgb = np.asarray(image.convert('RGB'))
    blue = ((rgb[:, :, 2] > 125) & (rgb[:, :, 2] > 1.25 * rgb[:, :, 1]) &
            (rgb[:, :, 1] > 70) & (rgb[:, :, 0] < 130)).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(blue, 8)
    if count <= 1 or max(stats[1:, cv2.CC_STAT_AREA]) < 20:
        return [0.] * 14
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    yy, xx = np.nonzero(labels == index)
    coords = np.stack((xx, yy), axis=1).astype(float)
    center = coords.mean(axis=0)
    cov = np.cov(coords.T) if len(coords) > 1 else np.eye(2)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, -1]
    box = cv2.minAreaRect(coords.astype(np.float32))
    x, y, w, h, area = stats[index]
    hull = cv2.convexHull(coords.astype(np.float32))
    hull_area = max(float(cv2.contourArea(hull)), 1.)
    return [1., center[0] / 224, center[1] / 224,
            w / 224, h / 224, area / (224 * 224), area / hull_area,
            eigvals[0] / (224 * 224), eigvals[1] / (224 * 224),
            principal[0], principal[1],
            box[1][0] / 224, box[1][1] / 224, box[2] / 90]


def features(row, crop):
    values = []
    for camera in CAMERAS:
        values.extend(shape_features(crop['images'][camera]))
    values.extend(row['state'][:7])
    values.extend(row['task_vector'][2:4])
    return values


def truth_yaw(row, tcp_tip, mount):
    tip = pose_matrix(row['state'][:7]) @ tcp_tip
    relative = Rotation.from_quat(row['observed_sc_tip_pose_opening_frame'][3:7])
    port = Rotation.from_matrix(tip[:3, :3]) * relative.inv()
    return (port * mount.inv()).as_euler('xyz')[2]


def stats(values):
    return {'count': len(values), 'median': float(np.median(values)) if len(values) else None,
            'p95': float(np.percentile(values, 95)) if len(values) else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--crops', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ridge-alpha', type=float, default=10.)
    args = p.parse_args()
    rows = [json.loads(x) for x in args.labels.open()]
    crops = [json.loads(x) for x in args.crops.open()]
    if [(r['trial'], r['frame']) for r in rows] != [(r['trial'], r['frame']) for r in crops]:
        raise ValueError('Identity mismatch')
    tcp_tip = matrix(json.loads(Path('configs/hierarchical_recovery/sc_tcp_tip_gazebo_tf_251.json').read_text())['tcp_to_sc_tip'])
    mount = Rotation.from_quat(json.loads(Path('configs/hierarchical_recovery/sc_board_to_port_training_fixed.json').read_text())['board_to_port_xyzw'])
    x = np.asarray([features(r, c) for r, c in zip(rows, crops)])
    y = np.asarray([truth_yaw(r, tcp_tip, mount) for r in rows])
    train = np.asarray([i for i, r in enumerate(rows) if r['split'] == 'train'])
    val = np.asarray([i for i, r in enumerate(rows) if r['split'] == 'validation'])
    train_scenes = {rows[i]['scene_sha256'] for i in train}
    val_scenes = {rows[i]['scene_sha256'] for i in val}
    if train_scenes & val_scenes or not len(train) or not len(val):
        raise ValueError('Empty or leaking grouped split')
    mean, scale = x[train].mean(0), x[train].std(0).clip(1e-3)
    x = np.clip((x - mean) / scale, -10, 10)
    design = np.column_stack((np.ones(len(x)), x))
    gram = design[train].T @ design[train]
    regularizer = np.eye(gram.shape[0]) * args.ridge_alpha
    regularizer[0, 0] = 0
    weights = np.linalg.solve(gram + regularizer, design[train].T @ y[train])
    prediction = design @ weights
    report = {'schema': 'sc_blue_shape_yaw_probe/v1', 'ridge_alpha': args.ridge_alpha,
              'runtime_inputs': 'RGB-selected crop silhouette, measured TCP, task one-hot',
              'training_scenes': len(train_scenes), 'validation_scenes': len(val_scenes),
              'by_split': {}}
    for split, indices in [('train', train), ('validation', val)]:
        near = [i for i in indices if abs(rows[i]['observed_sc_tip_pose_opening_frame'][2]) < .03]
        error = np.abs(np.rad2deg(prediction - y))
        report['by_split'][split] = {'near': stats(error[near]),
            'per_episode_near_p95_deg': {trial: stats([error[i] for i in near if rows[i]['trial'] == trial])['p95']
                                     for trial in sorted({rows[i]['trial'] for i in near})}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    np.savez(args.output.with_suffix('.npz'), weights=weights, mean=mean, scale=scale)
    print(json.dumps(report['by_split']['validation']))


if __name__ == '__main__':
    main()
