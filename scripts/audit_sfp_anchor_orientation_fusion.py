#!/usr/bin/env python3
"""Shadow-test whether a frozen RGB port orientation makes an SFP anchor safe to stop on."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from sfp_port_cascade import SfpPortCascade


CAMERAS = ('center', 'left', 'right')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(run, cascade):
    audit = Path(run['run'])
    proposals = [json.loads(line) for line in (audit / 'runtime_trace.jsonl').open()]
    proposals = [r for r in proposals if r.get('event') == 'proposal' and len(r.get('state', [])) == 32]
    fused = []
    for row in run['predictions']:
        if not row['visual_accept']:
            continue
        state_row = min(proposals, key=lambda p: abs(p['sim_time'] - row['sim_time_s']))
        if abs(state_row['sim_time'] - row['sim_time_s']) > .051:
            raise ValueError('Missing matching causal proposal')
        native = {c: cv2.cvtColor(cv2.imread(row['frame_paths'][c]), cv2.COLOR_BGR2RGB)
                  for c in CAMERAS}
        small = {c: cv2.resize(native[c], (288, 256), interpolation=cv2.INTER_AREA)
                 for c in CAMERAS}
        with torch.inference_mode():
            pose, diagnostic = cascade.predict(native, small, np.asarray(state_row['state']),
                                                np.asarray(state_row['task_vector']))
        agreement = (float(np.linalg.norm(pose[:3, 3] - row['predicted_port_base_m']) * 1000)
                     if pose is not None else None)
        fused.append(dict(time_s=row['sim_time_s'], pose_base=pose.tolist() if pose is not None else None,
                          position_agreement_mm=agreement,
                          use_for_anchor=bool(pose is not None and agreement <= 5),
                          cascade_reason=diagnostic.get('reason'),
                          reprojection_px=diagnostic.get('reprojection_px')))
    fused_by_time = {r['time_s']:r for r in fused}
    anchor = None
    first = None
    last = None
    previous_tcp = None
    travel_mm = 0.0
    decisions = []
    for row in run['predictions']:
        tcp = np.asarray(row['measured_tcp_base_m'])
        if previous_tcp is not None:
            travel_mm += float(np.linalg.norm(tcp - previous_tcp) * 1000)
        previous_tcp = tcp
        if anchor is not None and travel_mm > 150:
            anchor = None
            last = None
        fused_row = fused_by_time.get(row['sim_time_s'])
        if fused_row is not None and fused_row['use_for_anchor']:
            current = np.asarray(fused_row['pose_base'])
            if last is not None:
                distance = np.linalg.norm(current[:3, 3] - last[:3, 3]) * 1000
                angle = (Rotation.from_matrix(last[:3, :3]).inv() *
                         Rotation.from_matrix(current[:3, :3])).magnitude() * 180 / np.pi
                if row['sim_time_s'] - first <= 2 and distance <= 5 and angle <= 3:
                    anchor = current.copy()
                    anchor[:3, 3] = (last[:3, 3] + current[:3, 3]) / 2
                    travel_mm = 0.0
                elif distance > 5 or angle > 3 or row['sim_time_s'] - first > 2:
                    anchor = None
            last = current
            first = row['sim_time_s']
        predicted_axial = predicted_lateral = None
        if anchor is not None:
            relative = anchor[:3, :3].T @ (tcp - anchor[:3, 3])
            predicted_axial = float(relative[2] * 1000)
            predicted_lateral = float(np.linalg.norm(relative[:2]) * 1000)
        candidate = bool(anchor is not None and -100 <= predicted_axial <= -70 and
                         predicted_lateral <= 10)
        truth = bool(-100 <= row['true_axial_mm'] <= -70 and row['true_lateral_mm'] <= 10)
        decisions.append(dict(time_s=row['sim_time_s'], candidate_pause=candidate,
                              true_buffer=truth, predicted_axial_mm=predicted_axial,
                              predicted_lateral_mm=predicted_lateral,
                              true_axial_mm=row['true_axial_mm'],
                              true_lateral_mm=row['true_lateral_mm'],
                              anchor_active=anchor is not None))
    return dict(run=run['run'], accepted_position_only=run['accepted'],
                accepted_orientation_fusion=sum(r['use_for_anchor'] for r in fused),
                true_buffer_samples=sum(r['true_buffer'] for r in decisions),
                candidate_pauses=sum(r['candidate_pause'] for r in decisions),
                false_candidate_pauses=sum(r['candidate_pause'] and not r['true_buffer']
                                           for r in decisions),
                fused=fused, decisions=decisions)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--model-root', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve previous result')
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    root = a.model_root
    files = [root / 'joint_locator_native_sfp_v5_heatmap/best.pt',
             root / 'sfp_broad_crop480_v2/best.pt',
             root / 'sfp_fine_crop160_v2_inside_selection/best.pt',
             root / 'sfp_native_orientation_v2_highnative/best.pt',
             root / 'sfp_native_orientation_v3_runtime_matched/best.pt']
    model = SfpPortCascade(*files[:4], a.calibration, device=a.device,
                           orientation_path_secondary=files[4])
    source = json.loads(a.input.read_text())
    if source['schema'] != 'aic_sfp_selected_port_handoff_shadow/v1':
        p.error('Input schema mismatch')
    runs = [analyze(run, model) for run in source['runs']]
    result = dict(schema='aic_sfp_anchor_orientation_fusion_shadow/v1',
                  input=str(a.input), model_sha256={str(f):sha(f) for f in files},
                  true_geometry_runtime_input=False, insertion_actor_enabled=False,
                  final_scenes_opened=False, runs=runs)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps([{k:v for k,v in r.items() if k not in ('fused','decisions')}
                      for r in runs], indent=2))


if __name__ == '__main__':
    main()
