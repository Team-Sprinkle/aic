#!/usr/bin/env python3
"""Evaluate a causal 3D selected-port anchor on frozen SC identity predictions.

This is an offline diagnostic, never a runtime handoff. An anchor requires an
all-view identity accept and consistent observed pixels. Later rejected images
are rescued only if they agree geometrically with that anchor. Simulator
projection labels are used only for posthoc position error and proxy recall.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from evaluate_sc_native_triangulation import triangulate, reprojection_residual_px
from train_sc_native_keypoints import PretrainedKeypointRefiner

CAMERAS = ('center', 'left', 'right')


def pose(tcp):
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat(tcp[3:7]).as_matrix()
    out[:3, 3] = tcp[:3]
    return out


def current_geometry(row, optics):
    cameras = {cam: pose(row['tcp']) @ optics[cam] for cam in CAMERAS}
    selected = row.get('refined_pixels', row['requested_pixels'])
    pixels = {cam: np.asarray(selected[j], float)
              for j, cam in enumerate(CAMERAS)}
    if min(row['flags_requested']) < .5:
        return None
    try:
        point = triangulate(pixels, cameras)
        residual = reprojection_residual_px(point, pixels, cameras)
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not np.isfinite(point).all() or not np.isfinite(residual):
        return None
    return point, float(residual)


def refine(rows, checkpoint, device):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = PretrainedKeypointRefiner(None, False, point_count=1).to(device).eval()
    model.load_state_dict(saved['model'])
    samples = [(row, j) for row in rows for j in range(3)]
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    for start in range(0, len(samples), 48):
        batch = samples[start:start+48]
        crops, origins = [], []
        for row, j in batch:
            image = cv2.imread(row['images'][j])
            if image is None:
                raise FileNotFoundError(row['images'][j])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            left, top = np.round(np.asarray(row['requested_pixels'][j])-80).astype(int)
            crops.append(cv2.warpAffine(image, np.array([[1, 0, -left], [0, 1, -top]], np.float32),
                                        (160, 160), borderMode=cv2.BORDER_CONSTANT))
            origins.append((left, top))
        x = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).to(device).float()/255
        with torch.inference_mode():
            offsets = model(x).float().cpu().numpy().reshape(-1, 2)
        for (row, j), uv, origin in zip(batch, offsets, origins):
            if 'refined_pixels' not in row:
                row['refined_pixels'] = [None, None, None]
            refined = uv*159+origin
            row['refined_pixels'][j] = refined.tolist()
    for row in rows:
        jumps = [np.linalg.norm(np.asarray(f)-np.asarray(c)) for f, c in
                 zip(row['refined_pixels'], row['requested_pixels'])]
        if max(jumps) > 50:
            row.pop('refined_pixels')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('audits', nargs='+', type=Path)
    p.add_argument('--calibration', required=True, type=Path)
    p.add_argument('--refiner', type=Path)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--alternate-request', action='store_true',
                   help='For two-port scenes, audit the other physically present port')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--anchor-residual-px', type=float, default=2.)
    p.add_argument('--track-residual-px', type=float, default=2.)
    p.add_argument('--track-distance-mm', type=float, default=5.)
    p.add_argument('--max-travel-mm', type=float, default=150.)
    p.add_argument('--keyframe-spacing-mm', type=float, default=0.,
                   help='Minimum measured TCP travel between anchor refreshes')
    a = p.parse_args()
    if a.output.exists():
        p.error('Refusing to overwrite prior result')
    calibration = json.loads(a.calibration.read_text())
    optics = {cam: np.asarray(calibration['tcp_to_optical'][cam], float) for cam in CAMERAS}
    grouped = defaultdict(list)
    for audit in a.audits:
        for row in json.loads(audit.read_text())['triplets']:
            if a.alternate_request:
                row['requested_port'] = 1-row['requested_port']
                for left, right in (('scores_requested', 'scores_absent'),
                                    ('flags_requested', 'flags_absent'),
                                    ('requested_pixels', 'absent_pixels')):
                    row[left], row[right] = row[right], row[left]
            grouped[row['group']].append(row)
    if a.refiner:
        refine([row for rows in grouped.values() for row in rows], a.refiner, a.device)
    results = []
    metrics = {}
    for group, rows in sorted(grouped.items()):
        rows.sort(key=lambda x: x['sim_ns'])
        anchor = None
        anchor_tcp = None
        travel = 0.
        last_tcp = None
        keyframes = 0
        for row in rows:
            tcp = np.asarray(row['tcp'][:3], float)
            if last_tcp is not None:
                travel += 1000*np.linalg.norm(tcp-last_tcp)
            last_tcp = tcp
            geometry = current_geometry(row, optics)
            accepted = min(row['scores_requested']) >= .5
            eligible = geometry is not None and geometry[1] <= a.anchor_residual_px
            just_anchored = anchor is None and accepted and eligible
            if just_anchored:
                anchor, anchor_tcp, anchor_travel = geometry[0], tcp, travel
                keyframes += 1
            anchor_distance = (1000*np.linalg.norm(geometry[0]-anchor)
                               if geometry is not None and anchor is not None else None)
            # Refresh only from a new all-view identity accept that agrees
            # with the old 3D anchor. A rejected view cannot move the anchor.
            if (not just_anchored and anchor is not None and accepted and eligible and
                    anchor_distance <= a.track_distance_mm and
                    travel-anchor_travel >= a.keyframe_spacing_mm):
                anchor = .8*anchor+.2*geometry[0]
                anchor_travel = travel
                anchor_distance = 1000*np.linalg.norm(geometry[0]-anchor)
                keyframes += 1
            tracked = bool(anchor is not None and geometry is not None and
                           geometry[1] <= a.track_residual_px and
                           anchor_distance <= a.track_distance_mm and
                           travel-anchor_travel <= a.max_travel_mm)
            results.append({'group': group, 'sim_ns': row['sim_ns'],
                            'requested_port': row['requested_port'],
                            'projected_all_views': all(row['projected_inframe']),
                            'single_frame_identity': bool(accepted),
                            'causal_anchor_exists': anchor is not None,
                            'track_accept': tracked,
                            'residual_px': geometry[1] if geometry else None,
                            'track_distance_mm': anchor_distance,
                            'travel_since_anchor_mm': travel-anchor_travel if anchor is not None else None,
                            'anchor_base_m': anchor.tolist() if anchor is not None else None,
                            'base_opening_estimate_m': geometry[0].tolist() if geometry else None})
        group_rows = [x for x in results if x['group'] == group]
        positive = [x for x in group_rows if x['projected_all_views']]
        metrics[group] = {'samples': len(rows), 'projected_all_views': len(positive),
                          'keyframes': keyframes,
                          'single_frame_accept': sum(x['single_frame_identity'] for x in positive),
                          'tracked_accept': sum(x['track_accept'] for x in positive),
                          'first_anchor_ns': next((x['sim_ns'] for x in group_rows
                              if x['causal_anchor_exists']), None),
                          'last_track_accept': group_rows[-1]['track_accept']}
    result = {'schema': 'aic_sc_sparse_history_track_diagnostic/v1',
              'scope': 'offline diagnostic only, not a safe handoff',
              'limits': 'projected_all_views is not identifiable visibility; absent request cannot inherit a different request anchor',
              'inputs': [str(x) for x in a.audits],
              'alternate_request': a.alternate_request,
              'refiner': str(a.refiner) if a.refiner else None,
              'thresholds': {'anchor_residual_px': a.anchor_residual_px,
                             'track_residual_px': a.track_residual_px,
                             'track_distance_mm': a.track_distance_mm,
                             'max_travel_mm': a.max_travel_mm},
              'keyframe_spacing_mm': a.keyframe_spacing_mm,
              'metrics': metrics, 'rows': results}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
