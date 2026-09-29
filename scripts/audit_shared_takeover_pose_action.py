#!/usr/bin/env python3
"""Replay one scored SC takeover through a frozen shared actor (offline only)."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from train_shared_port_tcp_bc import SharedPortTCPActor, pose9


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--frames', type=Path, required=True)
    p.add_argument('--select', type=int, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    labels = [json.loads(line) for line in a.labels.open()]
    raw_frames = [json.loads(line) for line in a.frames.open()]
    nominal = {int(frame['frame']): float(frame['nominal_expert_time']) for frame in raw_frames}
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    use_tip = bool(ckpt.get('use_tip', False))
    keys = ['state_mean', 'state_std', 'pose_mean', 'pose_std', 'target_mean', 'target_std']
    if use_tip:
        keys += ['tip_mean', 'tip_std']
    stats = {key: ckpt['model'][key].numpy() for key in keys}
    model = SharedPortTCPActor(stats, pretrained=False, use_time=True,
                               use_tip=use_tip).to(a.device).eval()
    model.load_state_dict(ckpt['model'], strict=True)
    report = []
    for requested in a.select:
        row = min(labels, key=lambda item: abs(int(item['frame']) - requested))
        images = []
        for name in ('center', 'left', 'right'):
            with Image.open(row['images'][name]) as image:
                rgb = image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
                images.append(torch.from_numpy(np.asarray(rgb).copy()).permute(2, 0, 1))
        tip = torch.tensor(pose9(row['observed_sc_tip_pose_opening_frame']),
                           dtype=torch.float32, device=a.device)[None] if use_tip else None
        # Mirror the training-time SC near-port gate. This is a privileged
        # offline upper-bound audit, never a deployable gating rule.
        tip_active = use_tip and abs(float(row['observed_tcp_pose_opening_frame'][2])) < .03
        tip_mask = torch.tensor([tip_active], device=a.device) if use_tip else None
        with torch.inference_mode():
            predicted, guessed = model(
                torch.stack(images)[None].to(a.device),
                torch.tensor(row['state'], dtype=torch.float32, device=a.device)[None],
                torch.tensor(row['task_vector'], dtype=torch.float32, device=a.device)[None],
                torch.tensor([nominal[int(row['frame'])]], device=a.device),
                physical_tip=tip, tip_mask=tip_mask)
            oracle_predicted, _ = model(
                torch.stack(images)[None].to(a.device),
                torch.tensor(row['state'], dtype=torch.float32, device=a.device)[None],
                torch.tensor(row['task_vector'], dtype=torch.float32, device=a.device)[None],
                torch.tensor([nominal[int(row['frame'])]], device=a.device),
                external_pose=torch.tensor(
                    pose9(row['observed_tcp_pose_opening_frame']),
                    dtype=torch.float32, device=a.device)[None],
                external_mask=torch.tensor([True], device=a.device),
                physical_tip=tip, tip_mask=tip_mask)
        predicted = predicted[0, 0].cpu().numpy()
        guessed = guessed[0].cpu().numpy()
        true_pose = pose9(row['observed_tcp_pose_opening_frame'])
        true_target = pose9(row['teacher_tcp_target_poses_opening_frame'][0])
        actor_delta = (predicted[:3] - guessed[:3]) * 1000
        teacher_delta = (true_target[:3] - true_pose[:3]) * 1000
        oracle_delta = (oracle_predicted[0, 0, :3].cpu().numpy() - true_pose[:3]) * 1000
        report.append({'requested_frame': requested, 'frame': int(row['frame']),
                       'nominal_time_s': nominal[int(row['frame'])],
                       'pose_error_mm': float(np.linalg.norm(guessed[:3] - true_pose[:3]) * 1000),
                       'target_error_mm': float(np.linalg.norm(predicted[:3] - true_target[:3]) * 1000),
                       'actor_port_delta_mm': actor_delta.tolist(),
                       'teacher_port_delta_mm': teacher_delta.tolist(),
                       'delta_error_mm': float(np.linalg.norm(actor_delta - teacher_delta)),
                       'training_only_true_physical_tip_input': tip_active,
                       'oracle_tcp_pose_substitution': {
                           'training_only': True, 'oracle_port_delta_mm': oracle_delta.tolist(),
                           'delta_error_mm': float(np.linalg.norm(oracle_delta - teacher_delta))}})
    result = {'schema': 'shared_takeover_pose_action_audit/v1',
              'privileged_labels_used_for_posthoc_audit_only': True,
              'checkpoint': str(a.checkpoint), 'labels': str(a.labels),
              'frames': str(a.frames), 'rows': report}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'rows': len(report), 'output': str(a.output)}))


if __name__ == '__main__':
    main()
