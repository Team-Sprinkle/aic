#!/usr/bin/env python3
"""Posthoc requested-port transport scoring; never fed back to the actor."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from build_two_family_port_projection_labels import sfp_ports


def score(run: Path, calibration: dict) -> dict:
    config = yaml.safe_load((run / 'eval_config.yaml').read_text())
    trial_id = next(iter(config['trials']))
    ports = {(card, port): transform for card, port, transform in sfp_ports(
        {'trial_yaml': str(run / 'eval_config.yaml'), 'trial_id': trial_id}, calibration)}
    proposals = [json.loads(line) for line in (run / 'runtime_trace.jsonl').open()]
    proposals = [row for row in proposals if row.get('event') == 'proposal'
                 and len(row.get('state', [])) == 32]
    if not proposals:
        raise ValueError(f'No actor proposals: {run}')
    task = np.asarray(proposals[0]['task_vector'])
    card, port_id = int(task[4:9].argmax()), int(task[2:4].argmax())
    opening = ports[card, port_id]
    measured = []
    for row in proposals:
        current_task = np.asarray(row['task_vector'])
        if int(current_task[4:9].argmax()) != card or int(current_task[2:4].argmax()) != port_id:
            raise ValueError(f'Request changed: {run}')
        tcp = np.asarray(row['state'][:3])
        relative = opening[:3, :3].T @ (tcp - opening[:3, 3]) * 1000
        measured.append(dict(sim_time_s=float(row['sim_time']), axial_mm=float(relative[2]),
                             lateral_mm=float(np.linalg.norm(relative[:2])),
                             inference_ms=float(row.get('inference_ms', float('nan')))))
    band = [row for row in measured if -100 <= row['axial_mm'] <= -70]
    valid = [row for row in band if row['lateral_mm'] <= 10]
    official = yaml.safe_load((run / 'results/scoring.yaml').read_text())
    trial = next(value for key, value in official.items() if key.startswith('trial_'))
    latency = np.asarray([row['inference_ms'] for row in measured])
    latency = latency[np.isfinite(latency)]
    return dict(run=str(run), requested_card=card, requested_port=port_id,
                actor_proposals=len(measured), safe_buffer_visit=bool(valid),
                safe_buffer_samples=len(valid),
                closest_lateral_in_axial_band_mm=(min(row['lateral_mm'] for row in band)
                                                  if band else None),
                axial_band_samples=len(band),
                official_tier3=float(trial.get('tier_3', {}).get('score', 0)),
                official_contact_message=trial.get('tier_2', {}).get('categories', {}).get('contacts', {}).get('message'),
                inference_p50_p95_p99_ms=(np.percentile(latency, [50, 95, 99]).tolist()
                                          if len(latency) else None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve existing score')
    calibration = json.loads(args.calibration.read_text())
    results = [score(run, calibration) for run in args.run]
    args.output.write_text(json.dumps(dict(schema='aic_route_diffusion_dev_score/v1',
                                           actor_privileged_inputs=False,
                                           final_scenes_opened=False, runs=results), indent=2) + '\n')
    for result in results:
        print(json.dumps(result))


if __name__ == '__main__':
    main()
