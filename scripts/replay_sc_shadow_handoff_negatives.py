#!/usr/bin/env python3
"""Replay the frozen SC verifier on saved development camera triplets.

The capture's JPEGs are a documented proxy for live native RGB; the recorded
controller TCP and wrench are used at each image timestamp. No scored geometry
is available to the verifier. The same frame is tested with both requests.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from sc_shadow_handoff_verifier import SCShadowHandoffVerifier

CAMERAS = ('center', 'left', 'right')


def nearest(rows, stamp):
    return min(rows, key=lambda row: abs(row[0] - stamp))


def replay(audit, verifier, all_inspections=False):
    trace = [json.loads(s) for s in (audit / 'runtime_trace.jsonl').open()]
    inspections = [r for r in trace if r.get('event') == 'shadow_handoff_inspection' and
                   (all_inspections or r['valid'])]
    if not inspections:
        raise ValueError(f'No accepted inspection in {audit}')
    stream = json.loads((audit / 'scored_stream.json').read_text())['streams']
    controllers = stream['/aic_controller/controller_state']
    wrenches = stream['/fts_broadcaster/wrench']
    frames = {}
    for row in csv.DictReader((audit / 'frame_manifest.csv').open()):
        if row['camera'] in CAMERAS:
            frames.setdefault(int(row['sim_ns']), {})[row['camera']] = audit / 'frames' / row['file']

    output = []
    for inspection in inspections:
        request = inspection['requested_port']
        # The early image is a real offscreen transport state. It is evaluated
        # with its own measured TCP and force, never with the arrival state.
        for case, stamp, wanted in (
                ('accepted_control', round(inspection['sim_time'] * 1e9), request),
                ('same_image_wrong_request', round(inspection['sim_time'] * 1e9), 1-request),
                ('early_transport', int(10e9), request)):
            frame_stamp = min((s for s, fs in frames.items() if len(fs) == 3),
                              key=lambda s: abs(s-stamp))
            images = {c: cv2.cvtColor(cv2.imread(str(frames[frame_stamp][c])),
                                             cv2.COLOR_BGR2RGB) for c in CAMERAS}
            controller_stamp, controller = nearest(controllers, frame_stamp)
            wrench_stamp, wrench = nearest(wrenches, frame_stamp)
            pose = controller['tcp_pose']
            force = wrench['wrench']['force']
            state = np.zeros(32, np.float32)
            state[:7] = [pose['position'][k] for k in 'xyz'] + [pose['orientation'][k] for k in 'xyzw']
            state[26:29] = [force[k] for k in 'xyz']
            report = verifier.inspect(images, state, wanted)
            output.append(dict(audit=str(audit), case=case,
                               source_request=request, requested_port=wanted,
                               image_sim_ns=frame_stamp,
                               controller_offset_ms=(controller_stamp-frame_stamp)/1e6,
                               wrench_offset_ms=(wrench_stamp-frame_stamp)/1e6,
                               recorded_live_reason=inspection['reason'],
                               jpeg_proxy=True, report=report))
    return output


def replay_hidden(review_path, projection_dirs, verifier):
    review = json.loads(review_path.read_text())['rows']
    states = {}
    for directory in projection_dirs:
        for line in (directory / 'rows.jsonl').open():
            row = json.loads(line)
            states[(row['episode_id'], int(row['sim_ns']), row['camera'])] = row
    output = []
    for item in review:
        state_rows = [states.get((item['episode'], int(item['sim_ns']), c)) for c in CAMERAS]
        if any(row is None for row in state_rows):
            raise ValueError(f'Missing a camera or causal TCP in hidden review {item}')
        images = {c: cv2.cvtColor(cv2.imread(state_rows[j]['image']), cv2.COLOR_BGR2RGB)
                  for j, c in enumerate(CAMERAS)}
        state = np.zeros(32, np.float32)
        state[:7] = state_rows[0]['measured_tcp_base_pose']
        # Force was not retained in this review's derived rows. An acceptance
        # is therefore a possible false accept; an earlier-stage rejection is
        # definitive. The force assumption cannot make a rejection look safe.
        state[26:29] = (0, 0, 20)
        report = verifier.inspect(images, state, item['target_port'])
        output.append(dict(case='provisionally_hidden', review_index=item['review_index'],
                           scene_group=item['episode'], requested_port=item['target_port'],
                           image_sim_ns=item['sim_ns'], jpeg_proxy=True,
                           assumed_force_n=20, report=report))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('audits', nargs='+', type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--identity', type=Path,
                   default=Path('/var/tmp/chmin_aic_20260918_act/shadow_handoff_20261003/sc_spatial_identity.pt'))
    p.add_argument('--all-inspections', action='store_true')
    p.add_argument('--identity-scale', type=int, choices=(1,2), default=1)
    p.add_argument('--crop-identity', type=Path,
                   help='Opt-in observation-selected original-RGB crop identity checkpoint')
    p.add_argument('--diagnose-identity-failure', action='store_true',
                   help='Compute guarded orientation candidate after identity rejection')
    p.add_argument('--hidden-review', type=Path)
    p.add_argument('--hidden-projections', type=Path, nargs='*', default=[])
    a = p.parse_args()
    if a.output.exists():
        p.error('Refusing to overwrite existing replay')
    root = Path(__file__).resolve().parents[1]
    verifier = SCShadowHandoffVerifier(
        root / 'outputs/experiments/2026-10-02_two_family_perception/joint_locator_v1/best.pt',
        a.identity,
        '/var/tmp/chmin_aic_20260918_act/shadow_handoff_20261003/sc_orientation_oneport.pt',
        root / 'outputs/experiments/2026-10-02_two_family_perception/sc_native_refiner_v1/best.pt',
        root / 'configs/hierarchical_recovery/sc_visual_standoff_20261001.json',
        root / 'configs/hierarchical_recovery/sc_board_to_port_training_fixed.json',
        root / 'outputs/experiments/2026-10-02_two_family_perception/local_sc_orientation_prior_v1.json',
        device=a.device,identity_scale=a.identity_scale,
        crop_identity_path=a.crop_identity,
        diagnose_identity_failure=a.diagnose_identity_failure)
    rows = [r for audit in a.audits for r in replay(audit, verifier,a.all_inspections)]
    if a.hidden_review:
        rows.extend(replay_hidden(a.hidden_review, a.hidden_projections, verifier))
    counts = {case: {'accepted': sum(r['report']['valid'] for r in rows if r['case'] == case),
                     'tested': sum(r['case'] == case for r in rows)}
              for case in ('accepted_control', 'same_image_wrong_request',
                           'early_transport', 'provisionally_hidden')}
    result = dict(schema='sc_shadow_saved_jpeg_hard_negative_replay/v1',
                  interpretation='JPEG proxy only; repeat any passing gate on native RGB',
                  geometry_runtime=False, model_weights_frozen=True,
                  identity_scale=a.identity_scale,
                  crop_identity=str(a.crop_identity) if a.crop_identity else None,
                  diagnose_identity_failure=a.diagnose_identity_failure,
                  counts=counts, rows=rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
