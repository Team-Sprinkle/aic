#!/usr/bin/env python3
"""Score a frozen requested-port visual handoff on saved autonomous runs.

No scored port geometry is used until every image-based decision has been made.
The runner never commands the robot or changes the recorded transport policy.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import yaml

from build_two_family_port_projection_labels import sfp_ports
from sfp_selected_port_observation import SelectedSfpPort


CAMERAS = ('center', 'left', 'right')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def proposal_rows(run, period_s):
    rows = sorted((json.loads(line) for line in (run / 'runtime_trace.jsonl').open()),
                  key=lambda r: r.get('sim_time', -1))
    rows = [r for r in rows if r.get('event') == 'proposal' and len(r.get('state', [])) == 32]
    selected = []
    next_time = 0.0
    for row in rows:
        if row['sim_time'] >= next_time:
            selected.append(row)
            next_time = row['sim_time'] + period_s
    return selected


def frames_for(run):
    records = list(csv.DictReader((run / 'frame_manifest.csv').open()))
    return {camera: sorted([(int(r['sim_ns']), run / 'frames' / r['file'])
                            for r in records if r['camera'] == camera])
            for camera in CAMERAS}


def frame_at(rows, desired_ns, max_age_ns=100_000_000):
    if not rows:
        return None
    stamps = np.array([r[0] for r in rows])
    eligible = np.flatnonzero(stamps <= desired_ns)
    if not len(eligible):
        return None
    index = int(eligible[-1])
    return rows[index][1] if desired_ns - stamps[index] <= max_age_ns else None


def run_shadow(run, model, calibration, period_s):
    cfg = yaml.safe_load((run / 'eval_config.yaml').read_text())
    trial_id = next(iter(cfg['trials']))
    # Only evaluated after the prediction list has been frozen below.
    scheduled = proposal_rows(run, period_s)
    paths = frames_for(run)
    predictions = []
    for row in scheduled:
        image_stamps = row['image_sim_ns']
        chosen = {c: frame_at(paths[c], int(image_stamps[c])) for c in CAMERAS}
        if any(path is None for path in chosen.values()):
            continue
        images = {c: cv2.imread(str(chosen[c]), cv2.IMREAD_COLOR) for c in CAMERAS}
        if any(im is None for im in images.values()):
            raise FileNotFoundError(chosen)
        task = np.asarray(row['task_vector'])
        card = int(np.argmax(task[4:9]))
        port = int(np.argmax(task[2:4]))
        state = np.asarray(row['state'], float)
        started = time.perf_counter()
        pred = model.predict_batch([(images, state[:7], card, port)])[0]
        if model.device.type == 'cuda':
            torch.cuda.synchronize(model.device)
        latency_ms = 1000 * (time.perf_counter() - started)
        estimated = pred['predicted_base_m']
        predicted_distance = float(np.linalg.norm(estimated - state[:3]) * 1000) if estimated is not None else None
        predictions.append(dict(sim_time_s=float(row['sim_time']), card=card, port=port,
                                measured_tcp_base_m=state[:3].tolist(),
                                predicted_port_base_m=estimated.tolist() if estimated is not None else None,
                                visual_accept=pred['accepted'], reprojection_px=pred['reprojection_px'],
                                view_probabilities=pred['view_probabilities'],
                                predicted_tcp_port_distance_mm=predicted_distance,
                                inference_ms=latency_ms, frame_paths={c:str(chosen[c]) for c in CAMERAS}))
    # Port geometry is loaded only after all visual decisions have completed.
    true_ports = {(card, port): pose for card, port, pose in sfp_ports(
        {'trial_yaml': str(run / 'eval_config.yaml'), 'trial_id': trial_id}, calibration)}
    for item in predictions:
        pose = true_ports[item['card'], item['port']]
        rotation = pose[:3, :3]
        relative = rotation.T @ (np.asarray(item['measured_tcp_base_m']) - pose[:3, 3])
        item['true_axial_mm'] = float(relative[2] * 1000)
        item['true_lateral_mm'] = float(np.linalg.norm(relative[:2]) * 1000)
        estimate = item['predicted_port_base_m']
        if estimate is not None:
            item['true_pose_error_mm'] = float(np.linalg.norm(np.asarray(estimate) - pose[:3, 3]) * 1000)
            nearest = min(true_ports, key=lambda key: np.linalg.norm(
                np.asarray(estimate) - true_ports[key][:3, 3]))
            item['nearest_port_scoring_only'] = list(nearest)
            item['wrong_requested_port_scoring_only'] = nearest != (item['card'], item['port'])
        else:
            item['true_pose_error_mm'] = None
            item['nearest_port_scoring_only'] = None
            item['wrong_requested_port_scoring_only'] = None
    accepted = [r for r in predictions if r['visual_accept']]
    latencies = [r['inference_ms'] for r in predictions]
    return dict(run=str(run), sampled=len(predictions), accepted=len(accepted),
                accepted_wrong_port=sum(r['wrong_requested_port_scoring_only'] for r in accepted),
                accepted_pose_error_p95_mm=float(np.percentile([r['true_pose_error_mm'] for r in accepted], 95)) if accepted else None,
                first_visual_accept_s=accepted[0]['sim_time_s'] if accepted else None,
                true_buffer_visits=sum(-100 <= r['true_axial_mm'] <= -70 and r['true_lateral_mm'] <= 10 for r in predictions),
                inference_p50_p95_p99_ms=np.percentile(latencies, [50, 95, 99]).tolist() if latencies else None,
                predictions=predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--board', type=Path, required=True)
    parser.add_argument('--crop', type=Path, required=True)
    parser.add_argument('--port-calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--period-s', type=float, default=0.5)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve previous result')
    if args.period_s < 0.2:
        parser.error('Bound sampling to at least 0.2 s')
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    model = SelectedSfpPort(args.board, args.crop, args.device)
    model.warmup()
    calibration = json.loads(args.port_calibration.read_text())
    results = [run_shadow(run, model, calibration, args.period_s) for run in args.run]
    summary = dict(schema='aic_sfp_selected_port_handoff_shadow/v1',
                   source_runs=[str(p) for p in args.run], sampling_period_s=args.period_s,
                   board_sha256=digest(args.board), crop_sha256=digest(args.crop),
                   port_calibration_sha256=digest(args.port_calibration),
                   runtime_true_port_geometry_input=False, insertion_actor_enabled=False,
                   final_scenes_opened=False, runs=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='predictions'} for r in results], indent=2))


if __name__ == '__main__':
    main()
