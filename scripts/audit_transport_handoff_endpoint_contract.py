#!/usr/bin/env python3
"""Frozen observation-only SC endpoint replay, followed by separate label scoring."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from sc_shadow_handoff_verifier import SCShadowHandoffVerifier

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ('center', 'left', 'right')


def matrix(pose):
    m = np.eye(4)
    m[:3, 3] = pose[:3]
    m[:3, :3] = Rotation.from_quat(pose[3:7]).as_matrix()
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--transport', type=Path, required=True)
    p.add_argument('--local', type=Path, required=True)
    p.add_argument('--episodes', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve existing results; choose a new output directory.')
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    selected = {name: {'transport': [], 'local': []} for name in a.episodes}
    for phase, path in [('transport', a.transport), ('local', a.local)]:
        for line in path.open():
            row = json.loads(line)
            if row['episode_id'] in selected:
                selected[row['episode_id']][phase].append(row)
    paths = [ROOT/'outputs/experiments/2026-10-02_two_family_perception/joint_locator_v1/best.pt',
             Path('/var/tmp/chmin_aic_20260918_act/shadow_handoff_20261003/sc_spatial_identity.pt'),
             Path('/var/tmp/chmin_aic_20260918_act/shadow_handoff_20261003/sc_orientation_oneport.pt'),
             ROOT/'outputs/experiments/2026-10-02_two_family_perception/sc_native_refiner_v1/best.pt',
             ROOT/'configs/hierarchical_recovery/sc_visual_standoff_20261001.json',
             ROOT/'configs/hierarchical_recovery/sc_board_to_port_training_fixed.json',
             ROOT/'outputs/experiments/2026-10-02_two_family_perception/local_sc_orientation_prior_v1.json']
    verifier = SCShadowHandoffVerifier(*paths, device=a.device)
    a.output.mkdir(parents=True)
    pending = []
    for name, phases in selected.items():
        if not phases['transport'] or not phases['local']:
            raise ValueError(f'Missing transport/local episode {name}')
        end = max(phases['transport'], key=lambda r: r['sim_time'])
        windows = [('transport_endpoint', end)]
        for seconds in (1., 3., 5., 7.):
            candidates = [r for r in phases['local'] if r['sim_time'] <= end['sim_time']+seconds]
            if candidates:
                row = max(candidates, key=lambda r: r['sim_time'])
                if row['frame'] not in [r['frame'] for _, r in windows]:
                    windows.append((f'local_plus_{seconds:g}s', row))
        for kind, row in windows:
            images = {}
            for c in CAMERAS:
                bgr = cv2.imread(row['images'][c])
                if bgr is None:
                    raise FileNotFoundError(row['images'][c])
                images[c] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            requested = int(np.argmax(row['task_vector'][2:4]))
            tic = time.perf_counter()
            # Only observed state, RGB and public requested-port identity enter inference.
            report = verifier.inspect(images, np.asarray(row['state'], np.float32), requested)
            elapsed = (time.perf_counter()-tic)*1000
            wrong = verifier.inspect(images, np.asarray(row['state'], np.float32), 1-requested)
            item = dict(episode=name, kind=kind, frame=row['frame'], sim_time=row['sim_time'],
                        split=row['split'], requested_port=requested, images=row['images'],
                        jpeg_proxy=True, native_shape=list(images['center'].shape),
                        report=report, other_request_report=wrong, inference_ms=elapsed,
                        runtime_geometry_input=False)
            for c in CAMERAS:
                shown = cv2.cvtColor(images[c], cv2.COLOR_RGB2BGR)
                point = report.get('refined_pixels', {}).get(c)
                if point is not None:
                    xy = tuple(np.round(point).astype(int))
                    cv2.drawMarker(shown, xy, (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
                cv2.putText(shown, f'{name} {kind} request {requested}: {report["reason"]}',
                            (16, 32), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2)
                cv2.imwrite(str(a.output/f'{name}_{kind}_{c}.jpg'), shown)
            pending.append((item, row))
            print(json.dumps({k:item[k] for k in ('episode','kind','requested_port')} |
                             {'reason':report['reason']}), flush=True)
    # Freeze prediction results before any target geometry is used for scoring.
    (a.output/'predictions.json').write_text(json.dumps([i for i, _ in pending], indent=2)+'\n')
    scored = []
    for item, row in pending:
        base_opening = matrix(row['state'][:7]) @ np.linalg.inv(matrix(row['observed_tcp_pose_opening_frame']))
        report = item['report']
        truth = dict(tcp_axial_mm=row['observed_tcp_pose_opening_frame'][2]*1000,
                     tcp_lateral_mm=float(np.linalg.norm(row['observed_tcp_pose_opening_frame'][:2])*1000),
                     tip_axial_mm=row['observed_sc_tip_pose_opening_frame'][2]*1000)
        if 'predicted_opening_base_m' in report:
            truth['opening_error_mm'] = float(np.linalg.norm(np.asarray(report['predicted_opening_base_m'])-base_opening[:3, 3])*1000)
            truth['orientation_error_deg'] = float((Rotation.from_matrix(base_opening[:3, :3]).inv()*Rotation.from_quat(report['predicted_port_quat_xyzw'])).magnitude()*180/np.pi)
        scored.append(item | {'posthoc_geometry':truth})
    summary = dict(schema='aic_transport_handoff_endpoint_contract/v1', rows=scored,
                   models={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                   reasons=dict(Counter(i['report']['reason'] for i in scored)),
                   final_scenes_opened=False, autonomous_rollouts=0,
                   source_note='Recorded JPEGs at native dimensions; not lossless live RGB. Other-request accepts are counterfactuals, not proof of wrong identity without scoring that request.')
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary['reasons']), flush=True)


if __name__ == '__main__':
    main()
