#!/usr/bin/env python3
"""Training-only reset at a recorded PUBLIC robot pose; preserve relative grasp.

The engine adds gripper_offset in WORLD axes and interprets cable RPY in WORLD,
so both must rotate with the robot's changed initial wrist orientation.
No per-scene port truth or IK oracle is used to choose this training viewpoint.
"""
import argparse
import copy
import json
from pathlib import Path
import re
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--home-state', type=Path, required=True)
    parser.add_argument('--view-state', type=Path, required=True)
    args = parser.parse_args()
    home = np.asarray(json.loads(args.home_state.read_text())['state'])
    view = np.asarray(json.loads(args.view_state.read_text())['state'])
    config = yaml.safe_load((args.scene/'eval_config.yaml').read_text())
    frame = json.loads(args.home_state.read_text())['world_to_base_quat_xyzw']
    world_base = Rotation.from_quat(frame).as_matrix()
    delta = world_base @ (Rotation.from_quat(view[3:7]).as_matrix() @
                         Rotation.from_quat(home[3:7]).as_matrix().T) @ world_base.T
    original = copy.deepcopy(config)
    joints = list(config['robot']['home_joint_positions'])
    if len(joints) != 6:
        raise ValueError('Expected six measured UR joints')
    config['robot']['home_joint_positions'] = dict(zip(joints, map(float, view[19:25])))
    robot = (args.scene/'ur_llb_start.xacro').read_text()
    for key, value in config['robot']['home_joint_positions'].items():
        robot, count = re.subn(rf'({re.escape(key)}=)(-?[0-9]+\.[0-9]+)',
                              rf'\g<1>{value:.7f}', robot, count=1)
        assert count == 1
    for trial in config['trials'].values():
        for cable in trial['scene']['cables'].values():
            pose = cable['pose']
            rpy = (Rotation.from_matrix(delta) * Rotation.from_euler(
                'xyz', [pose[k] for k in ['roll', 'pitch', 'yaw']])).as_euler('xyz')
            offset = delta @ np.asarray([pose['gripper_offset'][k] for k in 'xyz'])
            for key, value in zip(['roll', 'pitch', 'yaw'], rpy):
                pose[key] = float(value)
            for key, value in zip('xyz', offset):
                pose['gripper_offset'][key] = float(value)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'eval_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    (args.output/'ur_llb_start.xacro').write_text(robot)
    for name in ['world_audit.sdf', 'bridge_audit.yaml']:
        shutil.copyfile(args.scene/name, args.output/name)
    (args.output/'reset_provenance.json').write_text(json.dumps(dict(
        original_scene=str(args.scene), original_robot=original['robot'],
        home_public_state=str(args.home_state), view_public_state=str(args.view_state),
        world_rotation_delta=delta.tolist(), pose_only=True, BC_expert=False,
        per_scene_target_geometry_used=False,
        physics_changed=False, cable_world_pose_rotated_to_preserve_grasp=True), indent=2)+'\n')


if __name__ == '__main__':
    main()
