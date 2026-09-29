#!/usr/bin/env python3
"""Time the complete four-model SC RGB perception path on retained images.

Includes native JPEG decode, frozen global locator, separate opening and
physical-tip crops, three refiner forwards, ray geometry, and board rails.
Excludes ROS transport, the shared actor, and controller command conversion.
No scored label, crop oracle, or scene TF enters a timed inference call.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch

from audit_sc_port_targets import pose_matrix
from build_sc_native_pose_labels import chain, WIDTH, HEIGHT
from evaluate_sc_native_triangulation import triangulate
from evaluate_sc_orientation_landmarks import POINTS, orientation
from evaluate_sc_board_lines_yaw import CAMERAS, camera_yaw
from train_sc_native_keypoints import PretrainedKeypointRefiner


def load_model(path, points, device):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    conditioned = saved.get('condition_target_port',
                            saved.get('architecture') == 'task_conditioned_resnet18_global_locator')
    model = PretrainedKeypointRefiner(None, conditioned,
                                     point_count=points).to(device).eval()
    model.load_state_dict(saved['model'], strict=True)
    return model


def tensor(images, device):
    return torch.stack([torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)
                        for image in images]).to(device).float() / 255


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--locator', type=Path, required=True)
    p.add_argument('--opening-refiner', type=Path, required=True)
    p.add_argument('--orientation-landmarks', type=Path, required=True)
    p.add_argument('--tip-refiner', type=Path, required=True)
    p.add_argument('--training-calibration-edges', type=Path, required=True)
    p.add_argument('--board-port-calibration-json', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=100)
    args = p.parse_args()
    torch.set_num_threads(8)
    rows = [json.loads(x) for x in args.labels.open()]
    source = json.loads(args.training_calibration_edges.read_text())['rows'][0]['edges']
    tool_tcp = chain(source, ['tool0', 'cam_mount/cam_mount_link', 'ati/base_link',
                              'ati/tool_link', 'gripper/hande_base_link', 'gripper/tcp'])
    camera_offsets = {}
    for cam in CAMERAS:
        tool_optical = chain(source, ['tool0', 'cam_mount/cam_mount_link',
            f'{cam}_camera/camera_link', f'{cam}_camera/sensor_link', f'{cam}_camera/optical'])
        camera_offsets[cam] = np.linalg.inv(tool_tcp) @ tool_optical
    mount = Rotation.from_quat(json.loads(args.board_port_calibration_json.read_text())['board_to_port_xyzw'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    locator = load_model(args.locator, 2, device)
    opening = load_model(args.opening_refiner, 2, device)
    landmarks = load_model(args.orientation_landmarks, len(POINTS), device)
    tip = load_model(args.tip_refiner, 1, device)
    indices = np.linspace(0, len(rows) - 1, min(args.samples, len(rows))).astype(int)
    times = []
    with torch.inference_mode():
        for index in indices:
            row = rows[index]
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            images = {}
            for cam in CAMERAS:
                with Image.open(row['images'][cam]) as source_image:
                    images[cam] = source_image.convert('RGB')
            task = torch.tensor(row['task_vector'][2:4], dtype=torch.float32,
                                device=device).repeat(3, 1)
            coarse = [images[cam].resize((288, 256), Image.Resampling.BILINEAR)
                      for cam in CAMERAS]
            global_pixel = locator(tensor(coarse, device), task).cpu().numpy().reshape(3, 2, 2)
            global_pixel *= np.array([WIDTH - 1, HEIGHT - 1])
            paired = []; plug = []; pair_origins = {}; tip_origins = {}
            for i, cam in enumerate(CAMERAS):
                pair_center = global_pixel[i].mean(0)
                tip_center = global_pixel[i, 0]
                pair_left, pair_top = round(pair_center[0] - 112), round(pair_center[1] - 112)
                tip_left, tip_top = round(tip_center[0] - 112), round(tip_center[1] - 112)
                pair_origins[cam] = np.array([pair_left, pair_top])
                tip_origins[cam] = np.array([tip_left, tip_top])
                paired.append(images[cam].crop((pair_left, pair_top, pair_left + 224, pair_top + 224)))
                plug.append(images[cam].crop((tip_left, tip_top, tip_left + 224, tip_top + 224)))
            pair_tensor = tensor(paired, device)
            tip_tensor = tensor(plug, device)
            opening_pixels = opening(pair_tensor, task).cpu().numpy().reshape(3, 2, 2) * 223
            landmark_pixels = landmarks(pair_tensor, task).cpu().numpy().reshape(3, len(POINTS), 2) * 223
            tip_pixels = tip(tip_tensor).cpu().numpy().reshape(3, 2) * 223
            cameras = {cam: pose_matrix(row['state'][:7]) @ camera_offsets[cam] for cam in CAMERAS}
            opening_xy = {cam: opening_pixels[i, 1] + pair_origins[cam]
                          for i, cam in enumerate(CAMERAS)}
            tip_xy = {cam: tip_pixels[i] + tip_origins[cam]
                      for i, cam in enumerate(CAMERAS)}
            landmark_xy = {cam: {name: landmark_pixels[i, j] + pair_origins[cam]
                                 for j, name in enumerate(POINTS)}
                           for i, cam in enumerate(CAMERAS)}
            _ = triangulate(opening_xy, cameras), triangulate(tip_xy, cameras)
            points = {name: triangulate({cam: landmark_xy[cam][name] for cam in CAMERAS}, cameras)
                      for name in POINTS}
            prior = (orientation(points) * mount.inv()).as_euler('xyz')[2]
            _ = [camera_yaw(row['images'][cam], opening_xy[cam], cameras[cam], prior)
                 for cam in CAMERAS]
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    measured = times[10:] if len(times) > 20 else times
    report = {'schema': 'sc_full_offline_perception_latency/v1',
              'scope': 'native JPEG decode, frozen RGB locator, paired and tip crops, opening/landmark/tip CNNs, triangulation and rail yaw; excludes ROS, actor, command conversion',
              'samples': len(times), 'warmup_excluded': len(times) - len(measured),
              'device': str(device), 'p50_ms': float(np.percentile(measured, 50)),
              'p95_ms': float(np.percentile(measured, 95)),
              'p99_ms': float(np.percentile(measured, 99))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
