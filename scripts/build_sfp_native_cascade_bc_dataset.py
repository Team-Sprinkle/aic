#!/usr/bin/env python3
"""Relabel scored native SFP commands into the frozen RGB-estimated port frame."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from audit_sc_port_targets import pose_matrix
from build_predicted_port_local_dataset import pose9
from sfp_port_cascade import SfpPortCascade


JOINT_ORDER = ('shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint')


def preceding(stream, t, tolerance_ns=100_000_000):
    i = bisect_right(stream, t, key=lambda row: row[0])-1
    if i < 0 or t-stream[i][0] > tolerance_ns:
        raise ValueError('Missing causal state sample')
    return stream[i][1]


def state32(stream, time_s):
    t = int(round(time_s*1e9))
    streams = stream['streams']
    ctrl = preceding(streams['/aic_controller/controller_state'], t)
    joints = preceding(streams['/joint_states'], t)
    wrench = preceding(streams['/fts_broadcaster/wrench'], t)['wrench']
    pose = ctrl['tcp_pose']
    velocities = ctrl['tcp_velocity']
    named = dict(zip(joints['name'], joints['position']))
    result = ([pose['position'][k] for k in 'xyz'] +
              [pose['orientation'][k] for k in 'xyzw'] +
              [velocities[d][k] for d in ('linear','angular') for k in 'xyz'] +
              list(ctrl['tcp_error']) + [named[k] for k in JOINT_ORDER] +
              [named['gripper/left_finger_joint']/2] +
              [wrench[d][k] for d in ('force','torque') for k in 'xyz'])
    if len(result) != 32 or not np.isfinite(result).all():
        raise ValueError('Invalid 32D state')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--coarse', type=Path, required=True)
    p.add_argument('--broad', type=Path, required=True)
    p.add_argument('--fine', type=Path, required=True)
    p.add_argument('--orientation', type=Path, required=True)
    p.add_argument('--orientation-secondary',type=Path)
    p.add_argument('--pair-orientation',action='store_true',
                   help='Use both RGB port openings to define the target frame')
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--selection-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sampling-boost', type=float, default=8.)
    p.add_argument('--max-supervision-position-error-mm',type=float,default=None,
                   help='Training-label-only exclusion of an RGB pose outside this 3D error')
    a = p.parse_args()
    if a.output.exists(): p.error('Preserve existing dataset')
    cv2.setNumThreads(1)
    import torch
    torch.set_num_threads(4)
    cascade = SfpPortCascade(a.coarse, a.broad, a.fine, a.orientation, a.calibration,
                             orientation_path_secondary=a.orientation_secondary,
                             pair_orientation=a.pair_orientation)
    selection = set(json.loads(a.selection_manifest.read_text())['selection_episodes'])
    untouched = set(json.loads(a.selection_manifest.read_text())['untouched_test_episodes'])
    sources = defaultdict(list)
    for row in map(json.loads, a.source.open()):
        sources[row['episode_id']].append(row)
    source_streams = {}
    for label in map(json.loads,(a.labels/'rows.jsonl').open()):
        source_streams[label['episode']] = label['source_stream']
    totals = Counter(); errors = defaultdict(list); split_groups = defaultdict(set)
    a.output.mkdir(parents=True)
    with (a.output/'rows.jsonl').open('w') as output:
        for episode, rows in sources.items():
            stream = json.loads(Path(source_streams[episode]).read_text())
            split = 'test' if episode in untouched else 'validation' if episode in selection else 'train'
            split_groups[split].add(stream['scene_group_id'])
            for source in rows:
                totals[(split,'source')]+=1
                native = {c: cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                          for c,path in source['images'].items()}
                small = {}
                for camera,image in native.items():
                    reduced = cv2.resize(image,(288,256),interpolation=cv2.INTER_AREA)
                    _,encoded = cv2.imencode('.jpg',reduced[...,::-1],[cv2.IMWRITE_JPEG_QUALITY,95])
                    small[camera] = cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),
                                                 cv2.COLOR_BGR2RGB)
                estimate,diagnostic = cascade.predict(native,small,source['state'],source['task_vector'])
                if estimate is None:
                    totals[(split,diagnostic['reason'])]+=1
                    continue
                try: state = state32(stream,source['timestamp_s'])
                except (KeyError,ValueError):
                    totals[(split,'causal_state_missing')]+=1;continue
                measured = pose_matrix(state[:7]);true_port_tcp = pose_matrix(
                    source['observed_tcp_pose_port_frame'])
                true_base_port = measured@np.linalg.inv(true_port_tcp)
                relative = np.linalg.inv(estimate)@measured
                teacher = [pose9(np.linalg.inv(estimate)@true_base_port@pose_matrix(target))
                           for target in source['teacher_tcp_target_poses_port_frame']]
                if len(teacher)!=4:raise ValueError('Four actual recorded command targets required')
                local_error = true_base_port[:3,:3].T@(estimate[:3,3]-true_base_port[:3,3])*1000
                if (a.max_supervision_position_error_mm is not None and
                    np.linalg.norm(local_error)>a.max_supervision_position_error_mm):
                    totals[(split,'supervision_pose_outlier')]+=1
                    continue
                errors[split].append(float(np.linalg.norm(local_error[:2])))
                output.write(json.dumps({'schema':'aic_predicted_port_local_bc/v1',
                    'family':'sfp','episode':episode,'episode_index':source['episode_index'],
                    'scene':stream['scene_group_id'],'split':split,'frame':source['frame_index'],
                    'time_s':source['timestamp_s'],'images':source['images'],
                    'image_kind':'native_1152x1024_rgb_jpeg','cache_index':None,
                    'state':state,'task':source['task_vector'],
                    'predicted_port_tcp_pose9':pose9(relative),
                    'teacher_tcp_targets_predicted_port_pose9':teacher,
                    'sampling_boost':a.sampling_boost,
                    'supervision_only_true_tcp_axial_m':source['observed_tcp_pose_port_frame'][2],
                    'perception_reprojection_residual_px':diagnostic['reprojection_px'],
                    'perception_accepted_views':3,'true_geometry_actor_input':False,
                    'recorded_complete_targets':True,
                    'teacher_takeover_age_s':source.get('teacher_takeover_age_s'),
                    'onpolicy_teacher_match':(source.get('teacher_takeover_age_s') is not None and
                        source['teacher_takeover_age_s']<=2.)},separators=(',',':'))+'\n')
                totals[(split,'accepted')]+=1
    if any(split_groups[x]&split_groups[y] for x,y in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('Scene leakage')
    report={'schema':'sfp_native_cascade_local_bc_dataset/v1',
            'counts':{str(k):v for k,v in totals.items()},
            'episodes':{s:len(v) for s,v in split_groups.items()},
            'accepted_lateral_error_mm':{s:{'median':float(np.median(v)),
                'p95':float(np.quantile(v,.95))} for s,v in errors.items() if v},
            'source_actions_actual_recorded':True,'runtime_geometry_input':False,
            'truth_scoring_and_relabel_only':True,
            'max_supervision_position_error_mm':a.max_supervision_position_error_mm,
            'pair_orientation':a.pair_orientation,
            'reserved_final_opened':False}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
