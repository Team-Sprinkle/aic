#!/usr/bin/env python3
"""Freeze grouped SFP selected-port projection labels from scored teacher RGB.

Gazebo geometry is used only to write labels. Runtime inputs are the three
original camera images, measured TCP, camera identity, and requested port code.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit_sc_port_targets import pose_matrix
from build_sc_native_pose_labels import project
from build_two_family_port_projection_labels import CAMERAS, sfp_ports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--transport-labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stride-seconds', type=float, default=1.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; preserve previous manifest')
    repo = Path(__file__).resolve().parents[1]
    optics_json = json.loads((repo/'configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())
    sfp_cal = json.loads((repo/'configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json').read_text())
    optics = {cam: np.asarray(optics_json['tcp_to_optical'][cam], float) for cam in CAMERAS}
    episodes = {}; selected = {}; splits = {}; scene_splits = {}
    for line in args.transport_labels.open():
        row = json.loads(line)
        if not 1800000 <= row['episode_index'] < 2000000:
            continue
        key = row['episode_index']
        if key not in episodes:
            episodes[key] = []
            selected[key] = (row['selected_card'], row['selected_port'])
            splits[key] = row['split']
        if selected[key] != (row['selected_card'], row['selected_port']):
            raise ValueError('Task changed inside episode')
        if scene_splits.setdefault(row['scene_sha256'], row['split']) != row['split']:
            raise ValueError('Reset leakage')
        episodes[key].append(row)
    if len(episodes) != 60 or len(scene_splits) != 60:
        raise ValueError((len(episodes), len(scene_splits)))
    all_rows = []
    for key, trajectory in sorted(episodes.items()):
        trajectory.sort(key=lambda r: r['timestamp_s'])
        audit = Path(trajectory[0]['source_scored_stream']).parent
        trial = json.loads((audit/'scored_stream.json').read_text())['trial_id']
        ports = sfp_ports({'trial_yaml': str(audit/'eval_config.yaml'), 'trial_id': trial}, sfp_cal)
        if selected[key] not in {(c,p) for c,p,_ in ports}:
            raise ValueError('Selected port absent')
        last = -float('inf')
        for row in trajectory:
            if row['timestamp_s'] - last < args.stride_seconds:
                continue
            last = row['timestamp_s']
            tcp = pose_matrix(row['state'][:7]); targets=[]
            for card, port, transform in ports:
                targets.append({'identity': f'sfp_card_{card}_port_{port}',
                                'position_base_m': transform[:3,3].tolist(),
                                'orientation_base_xyzw': Rotation.from_matrix(transform[:3,:3]).as_quat().tolist(),
                                'cameras': {cam:project(transform,tcp@optics[cam]) for cam in CAMERAS}})
            if not all(Path(x).is_file() for x in row['images'].values()):
                raise FileNotFoundError(row['images'])
            all_rows.append({'family':'sfp','episode':row['episode_id'],
                             'episode_index':key,'scene_group':row['scene_sha256'],
                             'split':row['split'],'time_s':row['timestamp_s'],
                             'state':row['state'][:7],'images':row['images'],
                             'image_kind':'native_1152x1024_rgb_jpeg',
                             'targets':targets,
                             'selected_identity':f'sfp_card_{selected[key][0]}_port_{selected[key][1]}',
                             'card_count':len({c for c,_,_ in ports}),
                             'geometry_training_label_only':True,
                             'source_scored_stream':row['source_scored_stream']})
    args.output.mkdir(parents=True)
    with (args.output/'rows.jsonl').open('w') as handle:
        for row in all_rows:
            handle.write(json.dumps(row,separators=(',',':'))+'\n')
    summary={'schema':'sfp_balanced_anchor_labels/v1','rows':len(all_rows),
             'episodes':Counter(splits.values()),'scene_groups':Counter(scene_splits.values()),
             'all_original_rgb':True,'teacher_actions_as_model_input':False,
             'geometry_runtime_input':False,'final_scenes_opened':False,
             'source':str(args.transport_labels),'stride_seconds':args.stride_seconds}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,default=dict))


if __name__=='__main__': main()
