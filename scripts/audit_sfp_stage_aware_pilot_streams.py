#!/usr/bin/env python3
"""Apply the frozen stage-label audit to the three uncached takeover pilots."""

import argparse
import json
from pathlib import Path

import yaml

from audit_sfp_stage_aware_takeover_labels import score_one


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve earlier audit output')
    manifest = json.loads((a.pilot / 'summary.json').read_text())
    records = []
    for case in manifest['runs']:
        audit = Path(case['audit'])
        source = json.loads((audit / 'scored_stream.json').read_text())
        event = json.loads((audit / 'corrective_teacher_takeover.json').read_text())
        takeover = float(event['takeover_sim_time_s'])
        report = yaml.safe_load((audit / 'results/scoring.yaml').read_text())
        trial = next(value for key, value in report.items() if key.startswith('trial_'))
        categories = trial['tier_2']['categories']
        ctrl = source['streams']['/aic_controller/controller_state']
        boundary = min(ctrl, key=lambda item: abs(item[0] * 1e-9 - takeover))
        origin = boundary[1]['tcp_pose']['position']
        rows = [{'time_s': takeover - .2,
                 'state': [origin['x'], origin['y'], origin['z']],
                 'actor_prefix_context_only': True}]
        last = takeover - .2
        for stamp, message in ctrl:
            t = stamp * 1e-9
            if t < takeover + .2 or t > takeover + 10 or t - last < .19:
                continue
            last = t
            pos = message['tcp_pose']['position']
            quat = message['tcp_pose']['orientation']
            rows.append({'time_s': t, 'state': [pos['x'], pos['y'], pos['z'],
                                               quat['x'], quat['y'], quat['z'], quat['w']],
                         'corrective_teacher': True, 'frame': len(rows)})
        meta = {'episode_id': case['name'], 'source_scored_stream': str(audit / 'scored_stream.json'),
                'official_tier3': case['official_score']['tier3'],
                'official_contact_score': categories['contacts']['score'],
                'official_insertion_force_score': categories['insertion force']['score']}
        row = {'scene': case.get('scene', case.get('scene_digest')),
               'card_count': case['card_count'], 'request': case['request'],
               'original_pilot_valid_label': case.get('valid_label')}
        row.update(score_one(rows, meta))
        records.append(row)
    output = {'schema': 'aic_sfp_stage_aware_pilot_stream_audit/v1',
              'pilot': str(a.pilot), 'sampling': 'causal measured TCP approximately 200ms',
              'full_successes': sum(x['official_full_insertion'] for x in records),
              'full_success_stage_labels': sum(x['official_full_insertion'] and x['usable'] for x in records),
              'policy_training_allowed': False, 'records': records}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({k: v for k, v in output.items() if k != 'records'}))


if __name__ == '__main__':
    main()
