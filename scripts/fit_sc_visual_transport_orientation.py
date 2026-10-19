#!/usr/bin/env python3
"""Fit a fixed SC transport TCP orientation using training scenes only."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--edges', type=Path, required=True)
    p.add_argument('--plans-root', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--validation-cases', nargs='+', required=True)
    a = p.parse_args()
    edges = json.loads(a.edges.read_text())['rows']
    validation = set(a.validation_cases)
    orientations = {}
    for row in edges:
        case = row['trial']
        plans = list(a.plans_root.glob(f'*/plans/{case}_insert/moveit_joint_trajectory.json'))
        if len(plans) != 1:
            raise ValueError(f'Expected one insert plan for {case}: {plans}')
        pose = json.loads(plans[0].read_text())['waypoints'][-1]['tcp_pose']
        orientations[case] = Rotation.from_quat(pose['orientation_xyzw'])
    train = sorted(set(orientations)-validation)
    if len(train) != 6 or len(validation) != 2:
        raise ValueError('Expected six calibration scenes and two held-out scenes')
    # All training rotations are close; rotvec averaging is stable in this range.
    mean = Rotation.from_rotvec(np.mean([orientations[k].as_rotvec() for k in train], axis=0))
    errors = {k: float((mean.inv()*v).magnitude()*180/np.pi)
              for k, v in orientations.items()}
    calibration = json.loads(a.calibration.read_text())
    if calibration['schema'] != 'aic_sc_visual_standoff_calibration/v1':
        raise ValueError('Unexpected calibration schema')
    calibration['transport_tcp_orientation_xyzw_training_mean'] = mean.as_quat().tolist()
    calibration['orientation_fit_train_cases'] = train
    calibration['orientation_heldout_cases'] = sorted(validation)
    calibration['orientation_heldout_errors_deg_posthoc'] = {k: errors[k] for k in sorted(validation)}
    a.calibration.write_text(json.dumps(calibration, indent=2)+'\n')
    print(json.dumps({'train_cases': train, 'heldout_errors_deg':
                      {k: errors[k] for k in sorted(validation)}}, indent=2))


if __name__ == '__main__':
    main()
