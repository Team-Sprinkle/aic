#!/usr/bin/env python3
"""Summarize frozen autonomous SC transport runs and posthoc teacher-path gates."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml


def one(directory):
    trace = [json.loads(s) for s in (directory/'runtime_trace.jsonl').open()]
    proposals = [x for x in trace if x['event'] == 'proposal']
    commands = [x for x in trace if x['event'] == 'command']
    terminal = [x for x in trace if x['event'] == 'terminal_before_return'][-1]
    signal = [x for x in trace if x['event'] == 'visual_standoff_signal']
    activation = next((x['sim_time'] for x in signal if x['active']), None)
    previsual = [x for x in commands if activation is None or x['sim_time'] < activation]
    diagnostic = json.loads((directory/'teacher_path_diagnostic.json').read_text())
    plan = json.loads(Path(diagnostic['plan']).read_text())['waypoints'][-1]['tcp_pose']
    actual = Rotation.from_quat(terminal['state'][3:7])
    desired = Rotation.from_quat(plan['orientation_xyzw'])
    final_angle = (actual.inv()*desired).magnitude()*180/np.pi
    latency = np.asarray([x['inference_ms'] for x in proposals])
    force = np.asarray([np.linalg.norm(x['state'][26:29]) for x in commands])
    scores = yaml.safe_load((directory/'results'/'scoring.yaml').read_text())
    trial = next(k for k in scores if k.startswith('trial_'))
    return {'run': directory.name,
            'case': json.loads((directory/'eval_manifest.json').read_text())['case'],
            'first_10mm_3deg_gate_time_s': diagnostic['first_10mm_3deg_gate_time_s'],
            'first_gate_position_mm': diagnostic['first_gate_position_mm'],
            'first_gate_orientation_deg': diagnostic['first_gate_orientation_deg'],
            'route_p95_before_gate_or_end_mm': diagnostic['route_deviation_before_first_gate_mm']['p95'],
            'terminal_position_error_mm': diagnostic['terminal_error_mm'],
            'terminal_orientation_error_deg': float(final_angle),
            'terminal_reason': terminal['reason'],
            'visual_first_activation_sim_s': activation,
            'minimum_previsual_tcp_z_mm': float(min(x['state'][2] for x in previsual)*1000),
            'transport_z_floor_guard_events': sum(x['event']=='sc_transport_z_floor_guard' for x in trace),
            'complete_inference_p50_p95_p99_ms': np.percentile(latency,[50,95,99]).tolist(),
            'stale_chunk_members_dropped': sum(x['event']=='stale_chunk_member_dropped' for x in trace),
            'raw_wrist_force_norm_p95_n': float(np.quantile(force,.95)),
            'official_tier_3_score': scores[trial]['tier_3']['score'],
            'geometry_input_at_runtime': False}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('runs',nargs='+',type=Path)
    a=p.parse_args()
    rows=[one(r) for r in a.runs]
    report={'schema':'aic_sc_visual_transport_development_comparison/v1',
            'runs':rows,'reserved_final_scenes_opened':False,
            'metric_scope':'posthoc teacher TCP transport standoff; no insertion actor in these runs'}
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
