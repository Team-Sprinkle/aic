#!/usr/bin/env python3
"""Label SFP CheatCode route phase and measured next waypoint on frozen RGB features."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from train_shared_port_tcp_bc import pose9


TRAIN_START = 1800000
VALIDATION_START = 1900000
INTERPOLATION_SECONDS = 5.5  # Frozen CheatCode.py interpolation_duration_sec
BOUNDARY_EXCLUSION_SECONDS = .15


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def relative_pose9(current, goal):
    current = np.asarray(current[:7], float)
    goal = np.asarray(goal[:7], float)
    rotation = Rotation.from_quat(current[3:]).inv()
    relative = rotation * Rotation.from_quat(goal[3:])
    return pose9(np.r_[rotation.apply(goal[:3] - current[:3]), relative.as_quat()]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sfp-labels', type=Path, required=True)
    parser.add_argument('--base-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve previous labeled cache')
    args.output.mkdir(parents=True)
    by_episode = defaultdict(list)
    for line in args.sfp_labels.open():
        row = json.loads(line)
        episode = int(row['episode_index'])
        if TRAIN_START <= episode < TRAIN_START + 50 or VALIDATION_START <= episode < VALIDATION_START + 10:
            by_episode[episode].append(row)
    if len(by_episode) != 60 or sum(TRAIN_START <= e < TRAIN_START + 50 for e in by_episode) != 50:
        raise ValueError(f'Expected 50 train and 10 development teacher episodes; found {len(by_episode)}')
    endpoints = {}
    episode_report = []
    groups = defaultdict(set)
    for episode, sequence in sorted(by_episode.items()):
        sequence.sort(key=lambda r: r['timestamp_s'])
        t0 = sequence[0]['timestamp_s']
        boundary = t0 + INTERPOLATION_SECONDS
        approach_end = min(sequence, key=lambda r: abs(r['timestamp_s'] - boundary))
        buffer_end = sequence[-1]
        if abs(approach_end['timestamp_s'] - boundary) > .12:
            raise ValueError(f'Missing approach-end measurement: {episode}')
        if buffer_end['timestamp_s'] - boundary < 2:
            raise ValueError(f'Descent too short for independent phase: {episode}')
        split = sequence[0]['split']
        if split != ('train' if episode < VALIDATION_START else 'validation'):
            raise ValueError(f'Episode split mismatch: {episode}')
        if len({r['scene_sha256'] for r in sequence}) != 1:
            raise ValueError(f'Scene changed during episode: {episode}')
        groups[split].add(sequence[0]['scene_sha256'])
        endpoints[episode] = dict(t0=t0, boundary=boundary,
                                  approach=np.asarray(approach_end['state'][:7], float),
                                  buffer=np.asarray(buffer_end['state'][:7], float))
        episode_report.append(dict(episode=episode, split=split,
                                   scene_sha256=sequence[0]['scene_sha256'],
                                   card=sequence[0]['selected_card'],
                                   port=sequence[0]['selected_port'],
                                   first_command_sim_s=t0,
                                   approach_end_sim_s=approach_end['timestamp_s'],
                                   buffer_end_sim_s=buffer_end['timestamp_s'],
                                   source_scored_stream=sequence[0]['source_scored_stream']))
    if groups['train'] & groups['validation']:
        raise ValueError('Complete-reset scene leakage')
    base_rows = [json.loads(line) for line in (args.base_cache / 'rows.jsonl').open()]
    old_features = np.load(args.base_cache / 'features.npy', mmap_mode='r')
    old_offsets = np.load(args.base_cache / 'camera_offsets.npy', mmap_mode='r')
    selected_indices = []
    output_rows = []
    stage_counts = Counter()
    for index, row in enumerate(base_rows):
        episode = int(row['episode'])
        if episode not in endpoints:
            continue
        endpoint = endpoints[episode]
        sim_time = row['time_s'] + endpoint['t0']
        # Cache time is elapsed since the first source row. Keep the policy's
        # 5.5 s phase boundary out of supervised examples to avoid edge noise.
        delta = sim_time - endpoint['boundary']
        if abs(delta) < BOUNDARY_EXCLUSION_SECONDS:
            continue
        stage = 0 if delta < 0 else 1
        goal = endpoint['approach' if stage == 0 else 'buffer']
        labeled = dict(row, stage=stage,
                       subgoal=relative_pose9(row['state'], goal),
                       handoff=0,route_label_source='measured future teacher TCP; never actor input',
                       source_feature_index=index)
        selected_indices.append(index)
        output_rows.append(labeled)
        stage_counts[(row['split'], stage)] += 1
    if len(output_rows) < 2000:
        raise ValueError('Unexpectedly small route-probe cache')
    for name, values in (('features.npy', old_features), ('camera_offsets.npy', old_offsets)):
        np.save(args.output / name, np.asarray(values[selected_indices]))
    with (args.output / 'rows.jsonl').open('w') as stream:
        for row in output_rows:
            stream.write(json.dumps(row, separators=(',', ':')) + '\n')
    manifest = dict(schema='aic_sfp_route_stage_probe_cache/v1',
                    teacher_policy='frozen stock CheatCode.py',
                    stage_names=['5.5-second approach interpolation', 'descent to buffered arrival'],
                    stage_boundary_from_source_code_s=INTERPOLATION_SECONDS,
                    boundary_exclusion_s=BOUNDARY_EXCLUSION_SECONDS,
                    waypoint='measured teacher TCP at end of current stage, expressed in current TCP body frame',
                    geometry_actor_input=False,teacher_stage_actor_input=False,
                    train_episodes=50,validation_episodes=10,
                    groups={key:len(value) for key,value in groups.items()},
                    counts={str(key):value for key,value in sorted(stage_counts.items())},
                    rows=len(output_rows),source_label_sha256=digest(args.sfp_labels),
                    source_cache_manifest_sha256=digest(args.base_cache / 'manifest.json'),
                    episodes=episode_report,final_scenes_opened=False)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(rows=len(output_rows),counts=manifest['counts'],groups=manifest['groups'])))


if __name__ == '__main__':
    main()
