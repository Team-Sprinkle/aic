#!/usr/bin/env python3
"""Match SC on-policy states to a scored teacher trajectory for diagnostic BC.

The teacher's complete recorded TCP target is expressed in the RGB-estimated
port frame. True plug/port TF is used only offline to match states by plug-tip
pose. Reject states far from the teacher manifold; no label is fabricated.
"""
import argparse
from bisect import bisect_right
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from analyze_local_actor_scored_tip import record_at
from audit_sc_port_targets import matrix, pose_matrix
from build_predicted_port_local_dataset import pose9
from shared_port_tcp_geometry import matrix_from_pose9


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rollout', type=Path, action='append', required=True)
    p.add_argument('--teacher', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--teacher-episode', default='llb_0001')
    p.add_argument('--period-s', type=float, default=0.4)
    p.add_argument('--max-tip-distance-mm', type=float, default=6)
    p.add_argument('--max-tip-orientation-deg', type=float, default=5)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve existing output')
    a.output.mkdir(parents=True)
    teachers = [r for r in map(json.loads, a.teacher.open()) if
                r.get('episode_id') == a.teacher_episode and
                r.get('official_tier3') == 75 and r.get('action_owner') == 'teacher']
    if not teachers:
        raise ValueError('No verified successful teacher rows')
    counters = {}
    kept = []
    for folder in a.rollout:
        scored = json.loads((folder / 'scored_stream.json').read_text())
        if scored['task']['target_module_name'] != 'sc_port_1':
            raise ValueError('Teacher episode and rollout port identity differ')
        episode = folder.name
        manifest = {}
        with (folder / 'frame_manifest.csv').open() as file:
            for row in csv.DictReader(file):
                manifest[(row['camera'], int(row['sim_ns']))] = str((folder / 'frames' / row['file']).resolve())
        trace = [json.loads(line) for line in (folder / 'local_actor_trace.jsonl').open()]
        perceptions = [r for r in trace if r['event'] == 'local_perception']
        proposals = [r for r in trace if r['event'] == 'proposal']
        if len(perceptions) != len(proposals):
            raise ValueError('Perception/proposal call count mismatch')
        edges = scored['edges']; port = 'task_board/sc_port_1'
        opening = (record_at(edges['aic_world|task_board'], 0) @
                   record_at(edges[f'task_board|{port}'], 0) @
                   record_at(edges[f'{port}|{port}/sc_port_base_link'], 0) @
                   matrix(json.loads(Path('configs/hierarchical_recovery/sc_port_opening_gazebo_tf_251.json').read_text())['base_to_opening']))
        tip_world = edges['aic_world|cable_1']; tip_link = edges['cable_1|cable_1/sc_tip_link']
        count = {'proposals': 0, 'kept': 0, 'missing_frame': 0, 'off_teacher_manifold': 0}
        last = -1e9
        for perception, proposal in zip(perceptions, proposals):
            t = float(proposal['sim_time'])
            if abs(float(perception['sim_time']) - t) > .02:
                raise ValueError(f'Perception/proposal clock mismatch at {t}')
            if t - last < a.period_s:
                continue
            last = t
            count['proposals'] += 1
            images = {camera: manifest.get((camera, int(proposal['image_sim_ns'][camera])))
                      for camera in ('center', 'left', 'right')}
            if not all(images.values()):
                count['missing_frame'] += 1
                continue
            sim_ns = int(round(t * 1e9))
            tip = np.linalg.inv(opening) @ record_at(tip_world, sim_ns) @ record_at(tip_link, sim_ns)
            tip_position = tip[:3, 3]
            candidates = []
            for teacher in teachers:
                target_tip = np.asarray(teacher['observed_sc_tip_pose_opening_frame'][:3], float)
                distance = float(np.linalg.norm(target_tip - tip_position) * 1000)
                teacher_rot = Rotation.from_quat(teacher['observed_sc_tip_pose_opening_frame'][3:7])
                angular = float((teacher_rot.inv() * Rotation.from_matrix(tip[:3, :3])).magnitude() * 180 / np.pi)
                candidates.append((distance + angular * .25, distance, angular, teacher))
            _, distance, angular, teacher = min(candidates, key=lambda x: x[0])
            if distance > a.max_tip_distance_mm or angular > a.max_tip_orientation_deg:
                count['off_teacher_manifold'] += 1
                continue
            state = proposal['state']
            predicted_port_tcp = perception['predicted_tcp_pose_in_port']
            estimated_base_port = pose_matrix(state[:7]) @ np.linalg.inv(matrix_from_pose9(predicted_port_tcp))
            converted = [pose9(np.linalg.inv(estimated_base_port) @ pose_matrix(target))
                         for target in teacher['teacher_tcp_target_poses_base_frame']]
            record = {
                'schema': 'aic_predicted_port_local_bc/v1', 'family': 'sc',
                'episode': episode, 'episode_index': -1, 'scene': scored['scene_group_id'],
                'split': 'train', 'frame': count['proposals'], 'time_s': t,
                'images': images, 'cache_index': None, 'state': state,
                'task': proposal['task_vector'],
                'predicted_port_tcp_pose9': predicted_port_tcp,
                'teacher_tcp_targets_predicted_port_pose9': converted,
                'image_kind': 'native_1152x1024_rgb_jpeg',
                'teacher_tip_match_distance_mm': distance,
                'teacher_tip_match_orientation_deg': angular,
                'teacher_source_time_s': teacher['sim_time'],
                'onpolicy_teacher_match': True,
                'true_geometry_actor_input': False,
                'recorded_complete_targets': True,
            }
            kept.append(record);count['kept'] += 1
        counters[episode] = count
    with (a.output / 'rows.jsonl').open('w') as file:
        for row in kept:
            file.write(json.dumps(row, separators=(',', ':')) + '\n')
    report = {'schema': 'aic_sc_onpolicy_teacher_match/v1', 'teacher_episode': a.teacher_episode,
              'teacher_rows': len(teachers), 'rollouts': counters,
              'max_tip_distance_mm': a.max_tip_distance_mm,
              'max_tip_orientation_deg': a.max_tip_orientation_deg,
              'period_s': a.period_s, 'runtime_geometry_input': False,
              'supervision_uses_scored_geometry': True, 'reserved_final_opened': False}
    (a.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
