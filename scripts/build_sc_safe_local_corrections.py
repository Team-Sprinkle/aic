#!/usr/bin/env python3
"""Select short, force-gated teacher targets on failed SC actor states.

These are local supervised correction labels, never successful expert episodes.
Ground-truth plug geometry is used only to select training rows and audit them.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--window-s', type=float, default=2.)
    p.add_argument('--max-force-rise-n', type=float, default=8.)
    p.add_argument('--absolute-force-ceiling-n', type=float, default=35.)
    p.add_argument('--max-handoff-baseline-force-n', type=float, default=35.)
    p.add_argument('--max-lateral-mm', type=float, default=10.)
    p.add_argument('--max-abs-axial-mm', type=float, default=50.)
    p.add_argument('--max-tcp-displacement-from-handoff-mm', type=float, default=25.,
                   help='Keep early corrections close to the measured teacher-established handoff TCP pose')
    a = p.parse_args()
    episodes = {int(x['episode_index']): x for x in json.loads(a.manifest.read_text())['episodes']}
    raw = {}
    starts = {}
    baseline_force = {}
    for episode, item in episodes.items():
        frames = [json.loads(line) for line in (Path(item['canonical_episode_dir']) / 'frames.jsonl').open()]
        raw[episode] = {int(frame['frame']): frame for frame in frames}
        active = [float(frame['nominal_expert_time']) for frame in frames if frame.get('student_active')]
        if not active:
            raise ValueError(f'No actor takeover in episode {episode}')
        starts[episode] = min(active)
        prehandoff = [float(np.linalg.norm(frame['state'][26:29])) for frame in frames
                      if starts[episode]-1 <= float(frame['nominal_expert_time']) < starts[episode]]
        if not prehandoff:
            raise ValueError(f'No pre-handoff force baseline in episode {episode}')
        baseline_force[episode] = float(np.median(prehandoff))
    all_rows = [json.loads(line) for line in a.labels.open()]
    handoff_tcp = {}
    for episode in episodes:
        episode_rows = [row for row in all_rows if int(row['episode_index']) == episode and
                        raw[episode].get(int(row['frame']), {}).get('student_active')]
        if not episode_rows:
            raise ValueError(f'No labeled actor-owned rows in episode {episode}')
        first = min(episode_rows, key=lambda row: int(row['frame']))
        handoff_tcp[episode] = np.asarray(first['observed_tcp_pose_opening_frame'][:3], dtype=float)
    selected = []
    exclusions = Counter()
    by_episode = Counter()
    gaps = []
    for row in all_rows:
        episode = int(row['episode_index'])
        frame = raw[episode].get(int(row['frame']))
        if frame is None:
            exclusions['missing_frame'] += 1; continue
        if not frame.get('student_active'):
            exclusions['teacher_owned'] += 1; continue
        if baseline_force[episode] > a.max_handoff_baseline_force_n:
            exclusions['episode_high_force_at_handoff'] += 1; continue
        nominal = float(frame['nominal_expert_time'])
        if not starts[episode] <= nominal < starts[episode] + a.window_s:
            exclusions['outside_early_window'] += 1; continue
        force = float(np.linalg.norm(frame['state'][26:29]))
        if (force >= baseline_force[episode] + a.max_force_rise_n or
                force >= a.absolute_force_ceiling_n):
            exclusions['force_over_gate'] += 1; continue
        tcp = np.asarray(row['observed_tcp_pose_opening_frame'][:3], dtype=float)
        tcp_displacement_mm = float(np.linalg.norm(tcp-handoff_tcp[episode])*1000)
        if tcp_displacement_mm > a.max_tcp_displacement_from_handoff_mm:
            exclusions['moved_beyond_handoff_corridor'] += 1; continue
        tip = np.asarray(row['observed_sc_tip_pose_opening_frame'][:3], dtype=float)
        if np.linalg.norm(tip[:2]) * 1000 > a.max_lateral_mm or abs(tip[2]) * 1000 > a.max_abs_axial_mm:
            exclusions['outside_near_port_gate'] += 1; continue
        proposal = np.asarray(frame['student_absolute_action'][:3], dtype=float)
        executed = np.asarray(frame['executed_target_pose'][:3], dtype=float)
        if np.linalg.norm(proposal-executed) > 1e-8:
            raise ValueError(f'Actor proposal not executed at episode {episode} frame {row["frame"]}')
        teacher = np.asarray(frame['teacher_target_pose'][:3], dtype=float)
        gaps.append(float(np.linalg.norm(teacher-executed)*1000))
        row['corrective_episode'] = True
        row['local_teacher_correction_from_failed_episode'] = True
        row['whole_episode_expert_success'] = False
        row['teacher_takeover_nominal_s'] = starts[episode]
        row['local_force_n'] = force
        row['local_tip_lateral_mm'] = float(np.linalg.norm(tip[:2])*1000)
        row['local_tip_axial_mm'] = float(tip[2]*1000)
        row['local_tcp_displacement_from_handoff_mm'] = tcp_displacement_mm
        selected.append(row)
        by_episode[episode] += 1
    if not selected:
        raise ValueError('No safe local correction rows selected')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('w') as stream:
        for row in selected:
            stream.write(json.dumps(row, separators=(',', ':'))+'\n')
    report = {'schema':'sc_safe_local_corrections/v1','training_only':True,
              'successful_expert_episodes':0,'whole_failed_episodes_admitted':0,
              'selected_chunks':len(selected),'by_episode':dict(by_episode),
              'excluded_reasons':dict(exclusions),
              'teacher_actor_target_gap_mm_median':float(np.median(gaps)),
              'teacher_actor_target_gap_mm_p95':float(np.quantile(gaps,.95)),
              'window_s':a.window_s,'max_force_rise_n':a.max_force_rise_n,
              'absolute_force_ceiling_n':a.absolute_force_ceiling_n,
              'max_handoff_baseline_force_n':a.max_handoff_baseline_force_n,
              'handoff_force_baseline_n':baseline_force,
              'max_lateral_mm':a.max_lateral_mm,'max_abs_axial_mm':a.max_abs_axial_mm,
              'max_tcp_displacement_from_handoff_mm':a.max_tcp_displacement_from_handoff_mm,
              'privileged_geometry_training_selection_only':True,
              'labels':str(a.output),'source_manifest':str(a.manifest)}
    a.summary.parent.mkdir(parents=True, exist_ok=True)
    a.summary.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('excluded_reasons',)}))


if __name__=='__main__': main()
