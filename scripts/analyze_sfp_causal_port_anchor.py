#!/usr/bin/env python3
"""Evaluate a causal selected-port anchor on frozen SFP shadow predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from build_two_family_port_projection_labels import sfp_ports


def analyze(run, tolerance_mm, travel_limit_mm, calibration,
            pair_travel_limit_mm=None, pair_max_age_s=2.0):
    audit = Path(run['run'])
    cfg = yaml.safe_load((audit / 'eval_config.yaml').read_text())
    trial_id = next(iter(cfg['trials']))
    truth = {(card, port): matrix[:3, 3] for card, port, matrix in sfp_ports(
        {'trial_yaml': str(audit / 'eval_config.yaml'), 'trial_id': trial_id}, calibration)}
    anchor = None
    first = None
    previous = None
    travel_since_refresh = 0.0
    consecutive = []
    last_visual_time = None
    last_visual_travel_mm = None
    total_travel_mm = 0.0
    rows = []
    for row in run['predictions']:
        tcp = np.asarray(row['measured_tcp_base_m'])
        if previous is not None:
            step_mm = float(np.linalg.norm(tcp - previous) * 1000)
            travel_since_refresh += step_mm
            total_travel_mm += step_mm
        previous = tcp
        if anchor is not None and travel_since_refresh > travel_limit_mm:
            anchor = None
            consecutive = []
        estimate = row['predicted_port_base_m']
        if row['visual_accept'] and estimate is not None:
            point = np.asarray(estimate)
            if last_visual_time is not None and pair_max_age_s > 0 and \
                    row['sim_time_s'] - last_visual_time > pair_max_age_s:
                consecutive = []
            if last_visual_travel_mm is not None and pair_travel_limit_mm is not None and \
                    total_travel_mm - last_visual_travel_mm > pair_travel_limit_mm:
                consecutive = []
            if consecutive and np.linalg.norm(point - consecutive[-1]) * 1000 > tolerance_mm:
                consecutive = []
            consecutive.append(point)
            last_visual_time = row['sim_time_s']
            last_visual_travel_mm = total_travel_mm
            if len(consecutive) >= 2:
                anchor = np.mean(consecutive[-2:], axis=0)
                travel_since_refresh = 0.0
                if first is None:
                    first = row['sim_time_s']
        valid = anchor is not None
        selected = (row['card'], row['port'])
        scoring_error = float(np.linalg.norm(anchor - truth[selected]) * 1000) if valid else None
        scoring_nearest = min(truth, key=lambda key: np.linalg.norm(anchor - truth[key])) if valid else None
        rows.append(dict(sim_time_s=row['sim_time_s'], anchor_valid=valid,
                         anchor_base_m=anchor.tolist() if valid else None,
                         traveled_since_refresh_mm=travel_since_refresh if valid else None,
                         anchor_error_mm_scoring_only=scoring_error,
                         nearest_port_scoring_only=list(scoring_nearest) if scoring_nearest else None,
                         true_buffer=bool(-100 <= row['true_axial_mm'] <= -70 and
                                          row['true_lateral_mm'] <= 10),
                         current_visual_accept=row['visual_accept'],
                         request=[row['card'], row['port']]))
    buffer = [r for r in rows if r['true_buffer']]
    return dict(run=run['run'], first_anchor_s=first,
                current_visual_accept_at_true_buffer=sum(r['current_visual_accept'] for r in buffer),
                anchored_at_true_buffer=sum(r['anchor_valid'] for r in buffer),
                anchor_error_at_true_buffer_mm=[r['anchor_error_mm_scoring_only']
                                                for r in buffer if r['anchor_valid']],
                wrong_anchor_at_true_buffer=sum(r['nearest_port_scoring_only'] != r['request']
                                                for r in buffer if r['anchor_valid']),
                true_buffer_samples=len(buffer), rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--pair-tolerance-mm', type=float, default=5)
    parser.add_argument('--travel-limit-mm', type=float, default=150)
    parser.add_argument('--pair-travel-limit-mm', type=float,
                        help='Maximum measured TCP path between accepted views')
    parser.add_argument('--pair-max-age-s', type=float, default=2.0,
                        help='Zero disables the elapsed-time limit; pair travel must then be set')
    parser.add_argument('--port-calibration', type=Path,
                        default=Path('configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve previous analysis')
    if args.pair_max_age_s <= 0 and args.pair_travel_limit_mm is None:
        parser.error('A pair must have a time or measured-travel limit')
    source = json.loads(args.input.read_text())
    if source['schema'] != 'aic_sfp_selected_port_handoff_shadow/v1':
        parser.error('Input schema mismatch')
    calibration = json.loads(args.port_calibration.read_text())
    result = dict(schema='aic_sfp_causal_selected_anchor/v1', input=str(args.input),
                  pair_tolerance_mm=args.pair_tolerance_mm,
                  travel_limit_mm=args.travel_limit_mm,
                  pair_travel_limit_mm=args.pair_travel_limit_mm,
                  pair_max_age_s=args.pair_max_age_s,
                  current_image_or_causal_history_only=True,
                  true_buffer_used_for_scoring_only=True,
                  final_scenes_opened=False,
                  runs=[analyze(run, args.pair_tolerance_mm, args.travel_limit_mm, calibration,
                                args.pair_travel_limit_mm,args.pair_max_age_s)
                        for run in source['runs']])
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps([{k:v for k,v in run.items() if k!='rows'} for run in result['runs']], indent=2))


if __name__ == '__main__':
    main()
