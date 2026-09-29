#!/usr/bin/env python3
"""Compare the shared actor's train and live camera resize paths on saved frames."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from train_shared_port_tcp_bc import SharedPortTCPActor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--task-vector', type=float, nargs=10, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-frames', type=int, default=64)
    a = p.parse_args()
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    weights = checkpoint['model']
    stats = {key: weights[key].numpy() for key in ('state_mean', 'state_std',
             'pose_mean', 'pose_std', 'target_mean', 'target_std')}
    model = SharedPortTCPActor(stats, pretrained=False,
                               use_time=bool(checkpoint['include_elapsed']))
    model.load_state_dict(weights, strict=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    frames = [json.loads(line) for line in (a.episode/'frames.jsonl').open()]
    first_time = frames[0]['sim_time']
    rows = []
    for frame in frames:
        if not frame.get('native_images'):
            continue
        train_views = []; live_views = []; two_stage_views = []; jpeg_views = []
        pixel_errors = []; two_stage_pixels = []; jpeg_pixels = []
        for camera in ('center', 'left', 'right'):
            with Image.open(a.episode/frame['images'][camera]) as f:
                train = np.asarray(f.convert('RGB').resize((224, 224),
                                   Image.Resampling.BILINEAR))
            with Image.open(a.episode/frame['native_images'][camera]) as f:
                native = np.asarray(f.convert('RGB'))
            live = cv2.resize(native, (224, 224), interpolation=cv2.INTER_AREA)
            intermediate = cv2.resize(native, (288, 256), interpolation=cv2.INTER_AREA)
            two_stage = np.asarray(Image.fromarray(intermediate).resize(
                (224, 224), Image.Resampling.BILINEAR))
            success, encoded = cv2.imencode('.jpg', intermediate[..., ::-1],
                                             [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not success:
                raise RuntimeError('JPEG encoding failed')
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)[..., ::-1]
            jpeg = np.asarray(Image.fromarray(decoded).resize(
                (224, 224), Image.Resampling.BILINEAR))
            train_views.append(torch.from_numpy(train.copy()).permute(2, 0, 1))
            live_views.append(torch.from_numpy(live.copy()).permute(2, 0, 1))
            two_stage_views.append(torch.from_numpy(two_stage.copy()).permute(2, 0, 1))
            jpeg_views.append(torch.from_numpy(jpeg.copy()).permute(2, 0, 1))
            pixel_errors.append(float(np.abs(train.astype(float) - live).mean()))
            two_stage_pixels.append(float(np.abs(train.astype(float) - two_stage).mean()))
            jpeg_pixels.append(float(np.abs(train.astype(float) - jpeg).mean()))
        state = torch.tensor(frame['state'], dtype=torch.float32, device=device)[None]
        task = torch.tensor(a.task_vector, dtype=torch.float32, device=device)[None]
        elapsed = torch.tensor([max(0., frame['sim_time'] - first_time)],
                               dtype=torch.float32, device=device)
        with torch.inference_mode():
            train_target, train_pose = model(torch.stack(train_views)[None].to(device),
                                             state, task, elapsed)
            live_target, live_pose = model(torch.stack(live_views)[None].to(device),
                                           state, task, elapsed)
            two_stage_target, two_stage_pose = model(
                torch.stack(two_stage_views)[None].to(device), state, task, elapsed)
            jpeg_target, jpeg_pose = model(
                torch.stack(jpeg_views)[None].to(device), state, task, elapsed)
        rows.append({'frame': frame['frame'], 'pixel_mae_rgb': pixel_errors,
                     'two_stage_pixel_mae_rgb': two_stage_pixels,
                     'jpeg_pixel_mae_rgb': jpeg_pixels,
                     'target_translation_difference_mm': float(torch.linalg.vector_norm(
                         train_target[0, 0, :3]-live_target[0, 0, :3])*1000),
                     'pose_translation_difference_mm': float(torch.linalg.vector_norm(
                         train_pose[0, :3]-live_pose[0, :3])*1000),
                     'two_stage_target_difference_mm': float(torch.linalg.vector_norm(
                         train_target[0, 0, :3]-two_stage_target[0, 0, :3])*1000),
                     'two_stage_pose_difference_mm': float(torch.linalg.vector_norm(
                         train_pose[0, :3]-two_stage_pose[0, :3])*1000),
                     'jpeg_target_difference_mm': float(torch.linalg.vector_norm(
                         train_target[0, 0, :3]-jpeg_target[0, 0, :3])*1000),
                     'jpeg_pose_difference_mm': float(torch.linalg.vector_norm(
                         train_pose[0, :3]-jpeg_pose[0, :3])*1000)})
        if len(rows) >= a.max_frames:
            break
    if not rows:
        raise ValueError('No native and resized image pairs')
    target = np.array([r['target_translation_difference_mm'] for r in rows])
    pose = np.array([r['pose_translation_difference_mm'] for r in rows])
    pixels = np.array([r['pixel_mae_rgb'] for r in rows])
    two_stage_target = np.array([r['two_stage_target_difference_mm'] for r in rows])
    jpeg_target = np.array([r['jpeg_target_difference_mm'] for r in rows])
    two_stage_pixels = np.array([r['two_stage_pixel_mae_rgb'] for r in rows])
    jpeg_pixels = np.array([r['jpeg_pixel_mae_rgb'] for r in rows])
    report = {'schema':'shared_bc_camera_preprocessing_parity/v1',
              'episode':str(a.episode), 'checkpoint':str(a.checkpoint),
              'comparison':'saved native JPEG direct INTER_AREA versus saved 288x256 JPEG then PIL BILINEAR; exact raw ROS bytes unavailable',
              'frames':len(rows), 'pixel_mae_rgb_mean_by_camera':pixels.mean(0).tolist(),
              'two_stage_pixel_mae_rgb_mean_by_camera':two_stage_pixels.mean(0).tolist(),
              'jpeg_pixel_mae_rgb_mean_by_camera':jpeg_pixels.mean(0).tolist(),
              'target_translation_difference_mm_median_p95':[float(np.median(target)),float(np.quantile(target,.95))],
              'two_stage_target_difference_mm_median_p95':[float(np.median(two_stage_target)),float(np.quantile(two_stage_target,.95))],
              'jpeg_target_difference_mm_median_p95':[float(np.median(jpeg_target)),float(np.quantile(jpeg_target,.95))],
              'pose_translation_difference_mm_median_p95':[float(np.median(pose)),float(np.quantile(pose,.95))],
              'rows':rows}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('frames','pixel_mae_rgb_mean_by_camera',
          'target_translation_difference_mm_median_p95',
          'two_stage_target_difference_mm_median_p95',
          'jpeg_target_difference_mm_median_p95')}))


if __name__=='__main__':
    main()
