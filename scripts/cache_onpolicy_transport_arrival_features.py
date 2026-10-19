#!/usr/bin/env python3
"""Cache frozen causal SC transport embeddings and posthoc arrival labels.

The encoder sees only saved RGB, ordinary measured state, public task code and
causal motion history. Teacher route geometry is opened after inference solely
to make offline labels. Do not treat those labels as deployable measurements.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torchvision.models import resnet18

from compare_frozen_sc_transport_stops import frames, nearest, route_error_mm
from shared_bc_camera_preprocessing import transport_rgb_tensor
from train_temporal_transport_probe import TemporalTransport


CAMERAS = ('center', 'left', 'right')


def cache_audit(audit: Path, model: TemporalTransport, backbone: torch.nn.Module,
                manifest: dict, state_mean: np.ndarray, state_std: np.ndarray,
                masked: np.ndarray, device: str) -> list[dict]:
    trace = [json.loads(line) for line in (audit / 'runtime_trace.jsonl').open()]
    proposals = [row for row in trace if row['event'] == 'proposal']
    if not proposals:
        return []
    diagnostic = json.loads((audit / 'teacher_path_diagnostic.json').read_text())
    plan_path = Path(diagnostic['plan'])
    plan = json.loads(plan_path.read_text())['waypoints']
    path = np.asarray([row['tcp_pose']['position'] for row in plan], dtype=float)
    goal = plan[-1]['tcp_pose']
    goal_xyz = np.asarray(goal['position'], dtype=float)
    goal_rot = Rotation.from_quat(goal['orientation_xyzw'])
    episode = plan_path.parts[-2]
    frame_index = frames(audit)
    history = []
    result = []
    positions = []
    previous_state = None
    captured = []

    def capture(_module, inputs):
        captured.append(inputs[0].detach().float().cpu().numpy()[0])

    hook = model.handoff.register_forward_pre_hook(capture)
    try:
        for proposal in proposals:
            time_s = float(proposal['sim_time'])
            state = np.asarray(proposal['state'], dtype=np.float32)
            images = []
            for camera in CAMERAS:
                rgb = cv2.imread(str(nearest(frame_index[camera],
                                             proposal['image_sim_ns'][camera])))
                if rgb is None:
                    raise FileNotFoundError(f'{audit} {camera} {time_s}')
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                images.append(transport_rgb_tensor(rgb, native_jpeg_quality=86))
            x = torch.stack(images).to(device).float() / 255
            x = (x - x.new_tensor((.485, .456, .406))[None, :, None, None]) / \
                x.new_tensor((.229, .224, .225))[None, :, None, None]
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                feature = backbone(x).reshape(1536).float()
            normalized = (state - state_mean) / state_std
            normalized[masked] = 0
            task = np.asarray(proposal['task_vector'], dtype=np.float32)
            motion = np.zeros(6, dtype=np.float32)
            if previous_state is not None:
                motion[:3] = state[:3] - previous_state[:3]
                motion[3:] = (Rotation.from_quat(previous_state[3:7]).inv() *
                              Rotation.from_quat(state[3:7])).as_rotvec()
            previous_state = state
            stamps = np.asarray([proposal['image_sim_ns'][cam] for cam in CAMERAS]) / 1e9
            history.append(dict(time=time_s, feature=feature, motion=motion,
                                state_task=np.r_[normalized, task],
                                camera_clock=np.r_[np.maximum(0, time_s-stamps)/.05, 1.]))
            times = np.asarray([record['time'] for record in history])
            chosen = []
            for age in manifest['history_ages_s']:
                j = np.searchsorted(times, time_s-age, side='right')-1
                if j >= 0 and (age == 0 or time_s-times[j] <= 60) and j not in chosen:
                    chosen.append(j)
            selected = [history[j] for j in sorted(chosen)]
            ages = np.asarray([(time_s-row['time'])/4 for row in selected], np.float32)
            gaps = np.asarray([0.] + [(selected[j]['time']-selected[j-1]['time'])/4
                                       for j in range(1, len(selected))], np.float32)
            tensor = lambda values: torch.as_tensor(np.asarray(values), device=device).float()[None]
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                _, _, _, handoff = model(
                    torch.stack([row['feature'] for row in selected])[None],
                    torch.zeros((1, len(selected), 36), device=device),
                    tensor([row['motion'] for row in selected]), tensor(ages), tensor(gaps),
                    tensor([row['camera_clock'] for row in selected]),
                    torch.ones((1, len(selected)), dtype=torch.bool, device=device),
                    tensor([row['state_task'] for row in selected]))
            if len(captured) != len(result)+1:
                raise RuntimeError('Frozen head embedding capture mismatch')
            positions.append(state[:3].astype(float))
            position_mm = float(np.linalg.norm(state[:3]-goal_xyz)*1000)
            angle_deg = float((Rotation.from_quat(state[3:7]).inv()*goal_rot).magnitude()*180/np.pi)
            result.append(dict(audit=str(audit), episode=episode, sim_time_s=time_s,
                               embedding=captured[-1], original_probability=float(handoff.sigmoid()[0]),
                               position_mm=position_mm, angle_deg=angle_deg))
    finally:
        hook.remove()
    route = route_error_mm(np.asarray(positions), path)
    max_route = np.maximum.accumulate(route)
    for row, deviation, prefix_max in zip(result, route, max_route):
        row['route_mm'] = float(deviation)
        row['max_route_mm'] = float(prefix_max)
        row['coarse_safe'] = bool(row['position_mm'] <= 80 and row['angle_deg'] <= 10 and
                                  prefix_max <= 50)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--handoff', type=Path, required=True)
    parser.add_argument('--actor', type=Path, required=True)
    parser.add_argument('--imagenet', type=Path, required=True)
    parser.add_argument('--frozen-comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve previous result; choose a new output')
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    saved = torch.load(args.handoff, map_location='cpu', weights_only=False)
    manifest = saved['source_manifest']
    if not manifest['zero_previous_command']:
        raise ValueError('Expected zero-previous-command checkpoint')
    model = TemporalTransport(saved['model']['target_mean'].numpy(),
        saved['model']['target_std'].numpy(), raw_state_skip=True,
        feature_dim=1536, state_task_dim=42, route_conditioning=False).to(args.device).eval()
    model.load_state_dict(saved['model'])
    actor = torch.load(args.actor, map_location='cpu', weights_only=False)
    state_mean = actor['model']['state_mean'].numpy()
    state_std = actor['model']['state_std'].numpy()
    masked = np.asarray(actor['masked_state_indices'], dtype=int)
    backbone = resnet18(weights=None)
    backbone.fc = torch.nn.Identity()
    weights = torch.load(args.imagenet, map_location='cpu', weights_only=True)
    backbone.load_state_dict({key: value for key, value in weights.items()
                              if not key.startswith('fc.')})
    backbone = backbone.to(args.device).eval()
    frozen = json.loads(args.frozen_comparison.read_text())
    heldout = {Path(row['audit']).resolve() for row in frozen['episodes']}
    audit_paths = sorted({path.parent for path in args.root.rglob('teacher_path_diagnostic.json')
                          if (path.parent/'runtime_trace.jsonl').exists() and
                          (path.parent/'frame_manifest.csv').exists()})
    rows = []
    errors = []
    for index, audit in enumerate(audit_paths, 1):
        try:
            part = cache_audit(audit, model, backbone, manifest, state_mean,
                               state_std, masked, args.device)
            rows.extend(part)
            print(json.dumps(dict(index=index, total=len(audit_paths), audit=audit.name,
                                  rows=len(part), safe=sum(x['coarse_safe'] for x in part))), flush=True)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            errors.append(dict(audit=str(audit), error=str(exc)))
            print(json.dumps(errors[-1]), flush=True)
    if not rows:
        raise RuntimeError('No usable audited features')
    embeddings = np.stack([row.pop('embedding') for row in rows]).astype(np.float16)
    args.output.mkdir(parents=True)
    np.save(args.output/'embeddings.npy', embeddings)
    for row in rows:
        row['is_frozen_comparison'] = Path(row['audit']).resolve() in heldout
    with (args.output/'rows.jsonl').open('w') as file:
        for row in rows:
            file.write(json.dumps(row, separators=(',', ':'))+'\n')
    summary = dict(schema='aic_onpolicy_transport_arrival_features/v1',
                   rows=len(rows), audits=len({row['audit'] for row in rows}),
                   episode_groups=len({row['episode'] for row in rows}),
                   frozen_comparison_audits=sum(Path(a).resolve() in heldout for a in
                                                {row['audit'] for row in rows}),
                   counts={str(key):value for key,value in Counter(
                       (row['is_frozen_comparison'], row['coarse_safe'])
                       for row in rows).items()}, errors=errors,
                   checkpoint=str(args.handoff), actor=str(args.actor),
                   image_backbone=str(args.imagenet),
                   label='posthoc 80mm/10deg/50mm route invitation-to-inspect',
                   reserved_final_opened=False)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({key:value for key,value in summary.items() if key!='errors'}), flush=True)


if __name__ == '__main__':
    main()
