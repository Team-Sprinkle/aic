#!/usr/bin/env python3
"""Replay a candidate on causal frozen development images; labels score only."""
import cv2
cv2.setNumThreads(2)
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from native_connector_pose_observation import NativeConnectorPoseObservation
from sc_spatial_grasp_observation import SCSpatialGraspObservation

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--records', type=Path, required=True)
p.add_argument('--rotation', type=Path, required=True)
p.add_argument('--translation', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)
rotation = NativeConnectorPoseObservation(a.rotation, 'cpu')
position = SCSpatialGraspObservation(a.translation, 'cpu')
records = json.loads(a.records.read_text())
train = json.loads((Path(rotation.plan['source'])/'manifest.json').read_text())
train_episodes = {r['episode'] for r in train if r['split'] == 'train'}
train_scenes = {r['scene'] for r in train if r['split'] == 'train'}
assert not train_episodes & {r['episode'] for r in records}
assert not train_scenes & {r['scene'] for r in records}
results = []
for row in records:
    images = {c: cv2.cvtColor(cv2.imread(row['images'][c]), cv2.COLOR_BGR2RGB) for c in ['center', 'left', 'right']}
    t = np.asarray(position.predict(images, already_training_jpeg=True)['predicted_tip_position_tcp_m'])
    r = np.asarray(rotation.predict(images, already_training_jpeg=True,
                                   visual_tip_position_tcp_m=t)['orientation_for_diagnostic_only'])
    truth = np.asarray(row['truth_pose'])
    relative = truth[:3, :3].T@r
    results.append(dict(episode=row['episode'], phase=row['phase'], time_s=row['time_s'],
        translation_error_mm=float(np.linalg.norm(t-truth[:3, 3])*1000),
        orientation_error_deg=float(np.degrees(Rotation.from_matrix(relative).magnitude())),
        axial_tilt_error_deg=float(np.degrees(np.arccos(np.clip(relative[2, 2], -1, 1)))),
        predicted_rotation=r.tolist(), predicted_translation_m=t.tolist(),
        old_orientation_error_deg=row['orientation_error_deg']))
phases = {}
for phase in sorted({r['phase'] for r in results}):
    rows = [r for r in results if r['phase'] == phase]
    counts = Counter(r['episode'] for r in rows)
    weights = np.array([1/counts[r['episode']] for r in rows])
    metrics = {}
    for field in ['translation_error_mm', 'orientation_error_deg', 'axial_tilt_error_deg']:
        values = np.array([r[field] for r in rows]);order = np.argsort(values)
        metrics[field] = dict(p50_p95=np.percentile(values, [50, 95]).tolist(),
            episode_balanced_p50_p95=np.interp([.5, .95], np.cumsum(weights[order])/weights.sum(), values[order]).tolist())
    phases[phase] = dict(episodes=len(counts), samples=len(rows), metrics=metrics)
report = dict(phases=phases, results=results, rotation_checkpoint_sha256=hashlib.sha256(a.rotation.read_bytes()).hexdigest(),
    translation_checkpoint_sha256=hashlib.sha256(a.translation.read_bytes()).hexdigest(),
    records=str(a.records), train_episode_and_scene_disjoint=True, geometry_for_scoring_only=True,
    automatic_deployment=False, original_validation_metrics_not_replaced=True)
(a.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(phases), flush=True)
