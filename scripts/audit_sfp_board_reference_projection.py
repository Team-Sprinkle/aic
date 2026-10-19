#!/usr/bin/env python3
"""Audit whether the fixed task-board outline fits native SFP wrist RGB.

Board pose and CAD corner projections are posthoc feasibility labels only.
Projected-in-frame does not mean visible; inspect original RGB before a fit.
This script neither trains a model nor supplies geometry to an actor.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import yaml

from audit_sc_port_targets import matrix, pose_matrix
from build_sc_native_pose_labels import project
from build_sfp_port_tcp_waypoints import transform


CAMERAS = ('center', 'left', 'right')
# task_board_base_macro.xacro: outer collision envelope in board coordinates.
# These are conservative *candidate* outline landmarks, not verified visual
# mesh corners or positive visibility labels.
BOARD_CORNERS_XY_M = ((-.15, -.214), (.15, -.214), (-.15, .2125), (.15, .2125))


def source_scene(row):
    stream = row.get('source_scored_stream') or row.get('source_stream')
    if stream:
        trial_path = Path(stream).parent / 'eval_config.yaml'
    elif row.get('source_run'):
        trial_path = Path(row['source_run']) / 'eval_config.yaml'
    else:
        raise ValueError(f'No scene source for {row["episode"]}')
    if not trial_path.is_file():
        raise FileNotFoundError(trial_path)
    trial = next(iter(yaml.safe_load(trial_path.read_text())['trials'].values()))
    board = trial['scene']['task_board']
    bp = board['pose']
    occupied = [bool(board[f'nic_rail_{i}']['entity_present']) for i in range(5)]
    return bp, occupied, trial_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, action='append', required=True,
                   help='Native projection label directory; repeat for train additions')
    p.add_argument('--camera-calibration', type=Path, required=True)
    p.add_argument('--sfp-calibration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-state-image-age-ms', type=float, default=10.0)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve existing output')
    a.output.mkdir(parents=True)
    cam = json.loads(a.camera_calibration.read_text())
    optics = {c: np.asarray(cam['tcp_to_optical'][c], float) for c in CAMERAS}
    sfp = json.loads(a.sfp_calibration.read_text())
    base_world = np.linalg.inv(matrix(sfp['world_to_base']))
    mount = sfp['board_to_nic_mount']
    cache = {}
    records = []
    roles = defaultdict(set)
    counts = Counter()
    rejected = Counter()
    for source in a.source:
        for line in (source / 'rows.jsonl').open():
            row = json.loads(line)
            if row['family'] != 'sfp' or row['image_kind'] != 'native_1152x1024_rgb_jpeg':
                raise ValueError('Expected native SFP labels')
            stamps = [int(Path(row['images'][c]).stem.split('_')[-2]) for c in CAMERAS]
            age = max(abs(float(row['time_s']) * 1e9 - t) for t in stamps) / 1e6
            spread = (max(stamps) - min(stamps)) / 1e6
            if age > a.max_state_image_age_ms or spread > a.max_state_image_age_ms:
                rejected[row['split']] += 1
                continue
            path = (row.get('source_scored_stream') or row.get('source_stream')
                    or row.get('source_run'))
            if path not in cache:
                bp, occupied, trial_path = source_scene(row)
                board_world = transform([bp[k] for k in ('x', 'y', 'z')],
                                        [bp[k] for k in ('roll', 'pitch', 'yaw')])
                board_base = base_world @ board_world
                corners = []
                for x, y in BOARD_CORNERS_XY_M:
                    pose = np.eye(4)
                    pose[:3, 3] = (x, y, .012)
                    corners.append(board_base @ pose)
                rails = []
                for slot in range(5):
                    pose = np.eye(4)
                    pose[:3, 3] = (mount['x_nominal_m'],
                                   mount['y_rail0_m'] + slot * mount['y_step_per_rail_m'],
                                   mount['z_m'])
                    rails.append(board_base @ pose)
                cache[path] = (corners, rails, occupied, str(trial_path))
            corners, rails, occupied, trial_path = cache[path]
            tcp = pose_matrix(np.asarray(row['state'][:7], float))
            projections = {c: [project(q, tcp @ optics[c]) for q in corners] for c in CAMERAS}
            rail_projections = {c: [project(q, tcp @ optics[c]) for q in rails] for c in CAMERAS}
            corner_flags = {c: [bool(q[3]) for q in projections[c]] for c in CAMERAS}
            full_views = [c for c in CAMERAS if all(corner_flags[c])]
            # An end is in-frame if either outer-corner candidate at that end is.
            both_ends = [c for c in CAMERAS if any(corner_flags[c][:2])
                         and any(corner_flags[c][2:])]
            rail_end_views = [c for c in CAMERAS if rail_projections[c][0][3]
                              and rail_projections[c][4][3]]
            split = row['split']
            roles[split].add(row['scene_group'])
            counts[(split, 'rows')] += 1
            counts[(split, 'full_any')] += bool(full_views)
            counts[(split, 'full_two')] += len(full_views) >= 2
            counts[(split, 'both_ends_any')] += bool(both_ends)
            counts[(split, 'rail_ends_any')] += bool(rail_end_views)
            counts[(split, 'rail_ends_two')] += len(rail_end_views) >= 2
            records.append({'episode': row['episode'], 'scene_group': row['scene_group'],
                            'split': split, 'time_s': row['time_s'],
                            'selected_identity': row['selected_identity'],
                            'card_count': row['card_count'], 'images': row['images'],
                            'occupied_slots_training_label_only': occupied,
                            'autonomous_actor_diagnostic': bool(row.get('autonomous_actor')),
                            'trial_yaml_label_only': trial_path,
                            'corner_projections_training_label_only': projections,
                            'rail_projections_training_label_only': rail_projections,
                            'full_outline_cameras': full_views,
                            'both_ends_in_frame_cameras': both_ends,
                            'rail_ends_in_frame_cameras': rail_end_views,
                            'projection_is_not_visibility': True})
    if roles['train'] & roles['validation']:
        raise ValueError('Train/development reset overlap')
    by_episode = defaultdict(list)
    for row in records:
        by_episode[(row['split'], row['scene_group'])].append(row)
    episodes = []
    for (split, scene), rows in sorted(by_episode.items()):
        rows.sort(key=lambda r: r['time_s'])
        first = rows[0]['time_s']
        early = [r for r in rows if r['time_s'] - first <= 3]
        episodes.append({'split': split, 'scene_group': scene, 'episode': rows[0]['episode'],
                         'card_count': rows[0]['card_count'], 'rows': len(rows),
                         'early_rows': len(early),
                         'early_full_outline_any': any(r['full_outline_cameras'] for r in early),
                         'early_both_ends_any': any(r['both_ends_in_frame_cameras'] for r in early),
                         'early_rail_ends_any': any(r['rail_ends_in_frame_cameras'] for r in early)})
    with (a.output / 'rows.jsonl').open('w') as f:
        for row in records:
            f.write(json.dumps(row, separators=(',', ':')) + '\n')
    summary = {'schema': 'sfp_board_outline_projection_feasibility/v1',
               'candidate_corners_board_xy_m': BOARD_CORNERS_XY_M,
               'source_rows': len(records), 'scene_groups': {k: len(v) for k, v in roles.items()},
               'rejected_stale': dict(rejected),
               'counts': {f'{k[0]}_{k[1]}': v for k, v in counts.items()},
               'early_scene_full_outline_any': {s: sum(e['early_full_outline_any'] for e in episodes
                                                       if e['split'] == s) for s in roles},
               'early_scene_both_ends_any': {s: sum(e['early_both_ends_any'] for e in episodes
                                                   if e['split'] == s) for s in roles},
               'early_scene_rail_ends_any': {s: sum(e['early_rail_ends_any'] for e in episodes
                                                   if e['split'] == s) for s in roles},
               'episodes': episodes, 'projected_in_frame_is_not_visibility': True,
               'geometry_runtime_input': False, 'final_scenes_opened': False}
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ('source_rows', 'scene_groups',
          'counts', 'early_scene_full_outline_any', 'early_scene_both_ends_any',
          'early_scene_rail_ends_any')}))


if __name__ == '__main__':
    main()
