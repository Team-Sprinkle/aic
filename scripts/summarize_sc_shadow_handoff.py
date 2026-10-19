#!/usr/bin/env python3
"""Summarize a bounded SC shadow handoff without opening final scenes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def summarize(audit):
    rows = [json.loads(s) for s in (audit / 'runtime_trace.jsonl').open()]
    checks = [r for r in rows if r['event'] == 'shadow_handoff_inspection']
    proposals = [r for r in rows if r['event'] == 'proposal']
    terminals = [r for r in rows if r['event'] == 'terminal_before_return']
    score = yaml.safe_load((audit / 'results/scoring.yaml').read_text())
    trial = next(v for k, v in score.items() if str(k).startswith('trial_'))
    # A checker runs after the held observation, not within the preceding
    # action proposal. Keep the two latency populations explicit.
    def latency(values):
        if not values:
            return None
        return {k: float(np.percentile(values, p)) for k, p in
                [('p50', 50), ('p95', 95), ('p99', 99)]}
    return {'audit': str(audit.resolve()),
            'scene_config_sha256': hashlib.sha256(
                (audit/'eval_config.yaml').read_bytes()).hexdigest(),
            'proposal_count': len(proposals), 'inspection_count': len(checks),
            'inspections': [{k: r.get(k) for k in ('sim_time', 'reason', 'valid',
                  'measured_tcp_posture_error_deg', 'measured_tcp_standoff_error_mm',
                  'measured_force_norm_n', 'opening_reprojection_px',
                  'board_view_spread_deg', 'verification_ms')} for r in checks],
            'accepted': any(r.get('valid') for r in checks),
            'orientation_correction_count': sum(r['event'] == 'shadow_orientation_target'
                                                for r in rows),
            'terminal_reason': terminals[-1]['reason'] if terminals else None,
            'proposal_latency_ms': latency([r['inference_ms'] for r in proposals]),
            'inspection_latency_ms': latency([r['verification_ms'] for r in checks
                                              if r.get('verification_ms') is not None]),
            'max_measured_force_norm_n': float(max(np.linalg.norm(r['state'][26:29])
                                                  for r in proposals)),
            'official_tier3': trial['tier_3']['score'],
            'scored_contact_message': trial['tier_2']['categories']['contacts']['message'],
            'runtime_true_geometry': False,
            'transport_handoff_only': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('audits', nargs='+', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve prior summary')
    result = {'schema': 'aic_sc_shadow_handoff_comparison/v1',
              'audits': [summarize(x) for x in a.audits],
              'final_scenes_opened': False}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
