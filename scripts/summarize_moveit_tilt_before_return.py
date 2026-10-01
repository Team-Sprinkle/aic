#!/usr/bin/env python3
"""Freeze the scored tilt-before-return pilot, including negative controls."""

import hashlib
import json
from pathlib import Path
import sys

import yaml


RAW = Path('/var/tmp/chmin_aic_20260918_act')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main(destination):
    repo = Path(__file__).resolve().parents[1]
    rows = []
    runs = list(RAW.glob('moveit_sc_port1_tilt*20260930'))
    runs += list(RAW.glob('moveit_sc_port1_old_tilt_matched_new_gates*20260930'))
    for run in sorted(runs):
        score_file = run / 'results/scoring.yaml'
        score = yaml.safe_load(score_file.read_text()) if score_file.is_file() else None
        trial = next((value for key, value in score.items() if key.startswith('trial_')), None) if score else None
        trace = run / 'runtime_trace.jsonl'
        events = [json.loads(line) for line in trace.open()] if trace.is_file() else []
        gates = [event for event in events if event.get('event') == 'tracking_gate_checked']
        bags = list((run / 'results').glob('bag_trial_*/*.mcap'))
        video_name = {
            'moveit_sc_port1_tilt_hold_gate12_align12_servo_insert_retry1_20260930': 'port1_tilt_hold_success_video',
            'moveit_sc_port1_old_tilt_matched_new_gates_repeat1_20260930': 'port1_old_tilt_matched_failure_video',
        }.get(run.name)
        video = (repo / 'outputs/experiments/2026-09-30_moveit_tilt_before_return' /
                 video_name / 'all_views_20fps.mp4') if video_name else None
        rows.append({
            'run': run.name,
            'explicit_tilt_before_return': 'old_tilt_matched' not in run.name,
            'raw_dir': str(run),
            'scored': trial is not None,
            'score_total': score.get('total') if score else None,
            'tier3': trial['tier_3']['score'] if trial else None,
            'tier3_message': trial['tier_3']['message'] if trial else None,
            'scored_contact_message': trial['tier_2'].get('categories', {}).get('contacts', {}).get('message') if trial else None,
            'scored_force_message': trial['tier_2'].get('categories', {}).get('insertion force', {}).get('message') if trial else None,
            'trace_event_count': len(events),
            'final_trace_event': events[-1].get('event') if events else None,
            'contact_event_count': sum(e.get('event') == 'contact_detected' for e in events),
            'recovery_event_count': sum('recovery' in e.get('event', '') for e in events),
            'tracking_gates': [{
                'passed': e.get('tracking_gate_passed'),
                'lateral_mm': e.get('final_lateral_error_m', 0) * 1000,
                'speed_mm_s': e.get('final_tcp_speed_mps', 0) * 1000,
                'force_delta_n': e.get('force_delta_n'),
            } for e in gates],
            'files_sha256': {name: sha(run / name) for name in (
                'eval_config.yaml', 'plan_summary.json', 'moveit_joint_trajectory.json',
                'inputs_sha256.txt', 'runtime_trace.jsonl', 'results/scoring.yaml')},
            'scored_mcap': [{'path': str(path), 'bytes': path.stat().st_size,
                             'sha256': sha(path)} for path in bags],
            'review_video': {'path': str(video.relative_to(repo)), 'bytes': video.stat().st_size,
                             'sha256': sha(video)} if video and video.is_file() else None,
        })
    result = {
        'schema': 'aic_moveit_tilt_before_return/v1',
        'scene': 'five NIC cards, SC port 1, fixed development reset 1002003',
        'teacher_privileged_geometry': True,
        'actor_autonomous': False,
        'gpu': 4,
        'reserved_final_scenes_opened': False,
        'route_quality_gate': 'passed_on_fixed_scene_under_progress_not_zero_touch_criterion',
        'repeatable_full_insertion_gate': 'open_due_to_earlier_near_port_failures_and_missing_new_starts',
        'runs': rows,
        'code_sha256': {name: sha(repo / name) for name in (
            'scripts/plan_moveit_sc_fixed_route.py',
            'scripts/hold_moveit_route_segments.py',
            'scripts/analyze_moveit_tilt_bag.py',
            'scripts/run_moveit_sc_route_pilot.sh',
            'aic_teacher_official/aic_teacher_official/OfficialTeacherReplay.py')},
    }
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'runs': len(rows), 'scored': sum(r['scored'] for r in rows),
                      'full_insertions': sum(r['tier3'] == 75 for r in rows)}))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
