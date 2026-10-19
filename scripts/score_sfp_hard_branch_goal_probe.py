#!/usr/bin/env python3
"""Score frozen early-route probes on independent hard-card teacher pairs."""

import argparse
from bisect import bisect_right
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch
from torchvision.models import resnet18

from lerobot_robot_aic.task_encoding import encode_task_vector
from materialize_sfp_scored_transport import source_path
from shared_bc_camera_preprocessing import transport_rgb_tensor


CAMERAS = ('center', 'left', 'right')


def at(rows, seconds):
    stamp = int(seconds * 1e9)
    i = bisect_right(rows, stamp, key=lambda row: row[0]) - 1
    if i < 0 or stamp - rows[i][0] > 110_000_000:
        return None
    return rows[i][1]


def pose(item):
    if item is None:
        return None
    p = item['tcp_pose']
    return np.asarray([p['position'][k] for k in ('x', 'y', 'z')] +
                      [p['orientation'][k] for k in ('x', 'y', 'z', 'w')], dtype=float)


def ridge(model, x):
    return ((x - model['center']) / model['scale']) @ model['coefficients'] + model['target_mean']


def sample(audit, stream):
    start = min(c['sim_ns'] for c in stream['commands'] if c['kind'] == 'pose') / 1e9
    ctrl = stream['streams']['/aic_controller/controller_state']
    card = int(stream['task']['target_module_name'].rsplit('_', 1)[-1])
    port = int(stream['task']['port_name'].rsplit('_', 1)[-1])
    task = np.asarray(encode_task_vector(task_family='sfp_to_nic',
                      target_card_index=card, target_port_index=port), dtype=float)
    chosen = []
    for dt in np.arange(0, 1.61, .2):
        t = start + dt
        current, future = pose(at(ctrl, t)), pose(at(ctrl, t + 1))
        if current is None or future is None:
            continue
        images = []
        for camera in CAMERAS:
            source = stream['cameras'][camera]
            i = bisect_right(source, int(t * 1e9), key=lambda row: row[0]) - 1
            if i < 0 or t * 1e9 - source[i][0] > 110_000_000:
                break
            path = source_path(audit, source[i][1])
            with Image.open(path) as handle:
                rgb = np.asarray(handle.convert('RGB'))
            images.append(transport_rgb_tensor(rgb, native_jpeg_quality=86,
                                               native_already_encoded=True))
        if len(images) != 3:
            continue
        chosen.append({'dt_s': float(dt), 'pose': current, 'task': task,
                       'future_motion': future[:3] - current[:3],
                       'images': images})
    return chosen


def score(rows, model, visual_model, backbone, device):
    if not rows:
        return
    batch = torch.stack([img for row in rows for img in row['images']]).to(device).float() / 255
    mean = batch.new_tensor((.485, .456, .406))[None, :, None, None]
    std = batch.new_tensor((.229, .224, .225))[None, :, None, None]
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        features = backbone((batch - mean) / std).reshape(len(rows), 1536).float().cpu().numpy()
    for i, row in enumerate(rows):
        row['features'] = features[i]
        row['pose_only_prediction'] = ridge(model, np.concatenate([row['pose'], row['task']]))
        row['visual_prediction'] = ridge(visual_model, np.concatenate([features[i], row['pose'], row['task']]))


def aggregate(rows, key):
    if not rows:
        return None
    err = [np.linalg.norm(r[key] - r['future_motion']) * 1000 for r in rows]
    return {'rows': len(rows), 'error_mm_p50': float(np.median(err)),
            'error_mm_p95': float(np.percentile(err, 95))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gate', type=Path, required=True)
    p.add_argument('--pose-model', type=Path, required=True)
    p.add_argument('--visual-model', type=Path, required=True)
    p.add_argument('--imagenet-resnet', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:7')
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve existing score')
    plan = json.loads((a.gate / 'plan.json').read_text())
    progress = json.loads((a.gate / 'progress.json').read_text())
    if not plan['final_scenes_opened'] is False:
        p.error('Final scenes were opened')
    torch.set_num_threads(6)
    backbone = resnet18(weights=None)
    backbone.fc = torch.nn.Identity()
    weights = torch.load(a.imagenet_resnet, map_location='cpu', weights_only=True)
    backbone.load_state_dict({k: v for k, v in weights.items() if not k.startswith('fc.')}, strict=True)
    backbone = backbone.to(a.device).eval()
    pose_model = dict(np.load(a.pose_model))
    visual_model = dict(np.load(a.visual_model))
    all_pairs = []
    all_rows = []
    for spec in plan['pairs']:
        entry = progress['results'][spec['name']]
        if not entry.get('gate_passed'):
            continue
        sides = {}
        for tag in ('original', 'alternate'):
            audit = Path(entry['variants'][tag]['launch']['output'])
            stream = json.loads((audit / 'scored_stream.json').read_text())
            sides[tag] = sample(audit, stream)
            score(sides[tag], pose_model, visual_model, backbone, a.device)
            all_rows.extend(sides[tag])
        for left in sides['original']:
            match = min(sides['alternate'], key=lambda r: abs(r['dt_s'] - left['dt_s']))
            if abs(left['dt_s'] - match['dt_s']) > .01:
                continue
            tcp_gap = np.linalg.norm(left['pose'][:3] - match['pose'][:3]) * 1000
            orientation_gap = (Rotation.from_quat(left['pose'][3:]).inv() *
                               Rotation.from_quat(match['pose'][3:])).magnitude() * 180 / np.pi
            cosine = float(np.dot(left['features'], match['features']) /
                           (np.linalg.norm(left['features']) * np.linalg.norm(match['features'])))
            if tcp_gap > 2 or orientation_gap > 2 or cosine < .995:
                continue
            truth = left['future_motion'] - match['future_motion']
            record = {'dt_s': left['dt_s'], 'true_branch_delta_mm': float(np.linalg.norm(truth) * 1000),
                      'tcp_gap_mm': float(tcp_gap), 'orientation_gap_deg': float(orientation_gap),
                      'visual_feature_cosine': cosine}
            for key, label in (('pose_only_prediction', 'pose'), ('visual_prediction', 'visual')):
                delta = left[key] - match[key]
                record[label + '_branch_delta_mm'] = float(np.linalg.norm(delta) * 1000)
                record[label + '_branch_error_mm'] = float(np.linalg.norm(delta - truth) * 1000)
                record[label + '_direction_cosine'] = float(np.dot(delta, truth) /
                    max(1e-9, np.linalg.norm(delta) * np.linalg.norm(truth)))
            all_pairs.append({'scene': spec['scene_digest'], 'card_count': spec['card_count'], **record})
    separable = [r for r in all_pairs if r['true_branch_delta_mm'] >= 10]
    result = {'schema': 'aic_sfp_hard_branch_frozen_goal_score/v1',
              'gate': str(a.gate), 'pairs_scored': len({r['scene'] for r in all_pairs}),
              'same_observation_rows': len(all_pairs),
              'separable_rows': len(separable),
              'separable_scenes': len({r['scene'] for r in separable}),
              'one_second_motion': {'pose': aggregate(all_rows, 'pose_only_prediction'),
                                    'visual': aggregate(all_rows, 'visual_prediction')},
              'branch': {kind: {'direction_fraction_cosine_ge_point_7': float(np.mean(
                                  [r[kind + '_direction_cosine'] >= .7 for r in separable]))
                                  if separable else None,
                                'error_mm_p50': float(np.median([r[kind + '_branch_error_mm'] for r in separable]))
                                if separable else None}
                         for kind in ('pose', 'visual')},
              'records': all_pairs, 'teacher_geometry_actor_input': False,
              'final_scenes_opened': False}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
