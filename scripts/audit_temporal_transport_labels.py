#!/usr/bin/env python3
"""Audit causal SC route-stage labels against recorded commands and plans.

Plans and opening geometry are used for offline labels/audit only. This script
does not produce a deployable input or modify the source dataset.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


STAGES = ('clear_start_side', 'shift_behind_row_low', 'traverse_beyond_row',
          'sweep_after_row', 'tilt_clear_of_cards', 'return_toward_port')


def read_jsonl(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def decode_task(vector):
    a = np.asarray(vector)
    assert a.shape == (10,) and np.all(np.isin(a, (0, 1)))
    assert a[:2].sum() == 1 and a[2:4].sum() == 1
    port = int(a[3] > .5)
    if a[0]:
        assert a[4:9].sum() == 1 and a[9] == 1
        return 'sfp', port, int(np.argmax(a[4:9]))
    assert a[4:].sum() == 0
    return 'sc', port, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', type=Path, required=True)
    ap.add_argument('--sc-rows', type=Path, required=True)
    ap.add_argument('--sfp-rows', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    spec = json.loads(args.spec.read_text())
    episodes = {e['case']: e for e in spec['episodes']}
    assert len(episodes) == 104
    grouped = defaultdict(list)
    for row in read_jsonl(args.sc_rows):
        grouped[row['episode_id']].append(row)
    assert set(grouped) == set(episodes)
    sfp_episodes = defaultdict(list)
    for row in read_jsonl(args.sfp_rows):
        sfp_episodes[(row['split'], row['episode_index'])].append(row)
    assert len(sfp_episodes) == 268
    sfp_scenes = defaultdict(set)
    sfp_counts = Counter()
    sfp_bad_task = []
    for (split, ep), rows in sfp_episodes.items():
        key = (rows[0]['scene_sha256'], tuple(rows[0]['task_vector']))
        sfp_scenes[split].add(key)
        family, port, card = decode_task(rows[0]['task_vector'])
        sfp_counts[(split, port, card)] += 1
        if family != 'sfp' or any(decode_task(r['task_vector']) != (family, port, card) for r in rows):
            sfp_bad_task.append(ep)
    reports = []
    sc_counts = Counter()
    all_distances = []
    all_camera_ms = []
    all_stage_rows = Counter()
    max_body_error_m = 0.0
    leakage = defaultdict(set)
    ambiguous = []
    for case, rows in sorted(grouped.items()):
        episode = episodes[case]
        assert {r['split'] for r in rows} == {episode['split']}
        assert {r['scene_sha256'] for r in rows} == {rows[0]['scene_sha256']}
        assert all(r['official_tier3'] == 75 for r in rows)
        leakage[episode['split']].add(episode['scene_reset_group'])
        family, port, card = decode_task(rows[0]['task_vector'])
        assert family == 'sc' and port == episode['selected_port'] and card is None
        assert all(decode_task(r['task_vector']) == (family, port, card) for r in rows)
        sc_counts[(episode['split'], episode['card_count'], port)] += 1
        plan_dir = Path(episode['insert_plan'])
        summary = json.loads((plan_dir / 'plan_summary.json').read_text())
        plan = json.loads((plan_dir / 'moveit_joint_trajectory.json').read_text())['waypoints']
        assert tuple(g['name'] for g in summary['candidate']['route_subgoals']) == STAGES
        assert all(a['timestamp'] < b['timestamp'] for a,b in zip(plan, plan[1:]))
        assert sum(w['diagnostics'].get('replay_modifier') == 'endpoint_joint_hold' for w in plan) == 2
        plan_start_s = float(rows[0]['episode_start_sim_time'])
        plan_times = [float(w['timestamp']) for w in plan]
        by_stage = Counter()
        distances = []
        camera_ms = []
        transition_examples = []
        previous_stage = None
        for r in rows:
            t = float(r['sim_time']) - plan_start_s
            # Time is an offline alignment key, never a stage input at rollout.
            j = min(max(bisect_right(plan_times, t)-1, 0), len(plan)-1)
            candidates = range(max(0,j-2), min(len(plan),j+4))
            base_target = np.asarray(r['teacher_tcp_target_poses_base_frame'][0][:3])
            nearest = min(candidates, key=lambda k: np.linalg.norm(
                base_target - np.asarray(plan[k]['tcp_pose']['position'])))
            d = float(np.linalg.norm(base_target - np.asarray(plan[nearest]['tcp_pose']['position'])))
            distances.append(d)
            stage = plan[j]['diagnostics']['segment']
            if stage != previous_stage:
                if previous_stage is not None:
                    transition_examples.append(dict(sim_time=r['sim_time'], from_stage=previous_stage,
                                                    to_stage=stage, image=r['images']['center'],
                                                    measured_tcp=r['state'][:7], target_distance_to_plan_m=d))
                previous_stage = stage
            by_stage[stage] += 1
            for cam in ('center','left','right'):
                camera_ms.append(abs(float(r['image_sim_ns'][cam])*1e-6 - float(r['sim_time'])*1e3))
            assert len(r['teacher_tcp_target_poses_body_frame']) == 4
            current = r['state'][:7]
            R = Rotation.from_quat(current[3:7]).as_matrix()
            for body, base in zip(r['teacher_tcp_target_poses_body_frame'],
                                  r['teacher_tcp_target_poses_base_frame']):
                reconstructed = np.asarray(current[:3]) + R @ np.asarray(body[:3])
                max_body_error_m = max(max_body_error_m, float(np.linalg.norm(
                    reconstructed - np.asarray(base[:3]))))
        assert set(by_stage) == set(STAGES), (case, by_stage)
        assert all(STAGES.index(b['to_stage']) == STAGES.index(b['from_stage'])+1
                   for b in transition_examples), (case, transition_examples)
        all_stage_rows.update(by_stage)
        all_distances.extend(distances)
        all_camera_ms.extend(camera_ms)
        if any(e['target_distance_to_plan_m'] > .02 for e in transition_examples):
            ambiguous.append(case)
        reports.append(dict(case=case, split=episode['split'], card_count=episode['card_count'],
                            selected_port=port, rows=len(rows), plan_duration_s=plan_times[-1],
                            route_stage_counts=dict(by_stage), commanded_plan_nearest_mm_p95=float(
                                np.quantile(distances,.95)*1e3),
                            commanded_plan_nearest_mm_max=float(max(distances)*1e3),
                            camera_abs_time_ms_max=float(max(camera_ms)),
                            local_start_after_plan_s=episode['local_start_sim_s']-
                                (plan_start_s+plan_times[-1]),
                            transition_examples=transition_examples))
    report = dict(schema='aic_temporal_transport_label_audit/v1',
                  sc_episodes=len(reports), sc_split=Counter(e['split'] for e in reports),
                  sc_coverage={str(k):v for k,v in sorted(sc_counts.items())},
                  sfp_episodes=len(sfp_episodes),sfp_split=Counter(k[0] for k in sfp_episodes),
                  sfp_coverage={str(k):v for k,v in sorted(sfp_counts.items())},
                  sfp_bad_task=sfp_bad_task,
                  sc_scene_reset_leakage=bool(leakage['train'] & leakage['validation']),
                  sfp_scene_task_leakage=bool(sfp_scenes['train'] & sfp_scenes['validation']),
                  stage_rows=dict(all_stage_rows),
                  command_plan_nearest_mm_p50=float(np.quantile(all_distances,.5)*1e3),
                  command_plan_nearest_mm_p95=float(np.quantile(all_distances,.95)*1e3),
                  command_plan_nearest_mm_max=float(max(all_distances)*1e3),
                  camera_abs_time_ms_p95=float(np.quantile(all_camera_ms,.95)),
                  camera_abs_time_ms_max=float(max(all_camera_ms)),
                  body_target_reconstruction_mm_max=max_body_error_m*1e3,
                  ambiguous_transition_cases=ambiguous,
                  sc_per_episode=reports,
                  labels_only='Plan stage/subgoal/port geometry are offline supervision; never runtime inputs',
                  final_scenes_opened=False,rl_started=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'sc_per_episode'},indent=2))


if __name__ == '__main__':
    main()
