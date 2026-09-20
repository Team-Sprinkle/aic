#!/usr/bin/env python3
"""Run the AIC Dreamer pilot through the shared, unchanged runtime evaluator.

The sole policy-name exception is for RunDreamerV4, which inherits the reviewed
RunACTTorchScript.insert_cable loop unchanged. Restore the exact common task and
simulation-duration audits after common preparation; other policies are rejected.
"""
from pathlib import Path
import math
import os
import sys
import yaml

import evaluate_act_checkpoints_runtime as shared

POLICY = 'aic_example_policies.ros.RunDreamerV4'
_BASE_PREPARE = shared.prepare_args
_BASE_SOURCES = shared.runtime_sources


def prepare_args(args):
    if args.policy_module != POLICY:
        raise ValueError('This evaluator is exclusively for the approved RunDreamerV4 inherited loop')
    if args.diagnostic_ground_truth or args.corrective_data_dir is not None:
        raise ValueError('Dreamer evaluation cannot access privileged geometry or collect teacher labels')
    if args.act_torchscript is not None:
        raise ValueError('Dreamer uses a control export with embedded normalization')
    if (args.max_simulation_sec, args.max_runtime_sec) != (90., 180.):
        raise ValueError('Dreamer comparison requires exactly90sim seconds and180wall watchdog')
    if (args.command_mode, args.command_frame, args.control_hz, args.control_clock,
            args.n_action_steps) != ('delta_pose', 'gripper/tcp', 20., 'simulation', 4) or args.delta_pose_reference != 'observation':
        raise ValueError('Dreamer requires observation-relative TCP delta20Hz execution of four commands')
    if args.temporal_ensemble_coeff is not None or args.start_delay_sec != 0:
        raise ValueError('Dreamer pilot has no temporal ensemble or artificial startup delay')
    if (args.max_translation_delta, args.max_rotation_delta, args.translation_limit_mode,
            args.translation_deadband, args.rotation_deadband) != (.1, .2, 'norm', 0., 0.):
        raise ValueError('Dreamer comparison must use the frozen physical command bounds')
    # Common prepare otherwise rejects the literal class name before any model or
    # simulator runs. All numeric/identity checks above are stricter, not disabled.
    simulation_limit = args.max_simulation_sec
    delta_reference = args.delta_pose_reference
    try:
        args.max_simulation_sec = None
        # The shared checker also restricts this inherited-loop option by the
        # literal ACT class name. Its semantics were checked strictly above.
        args.delta_pose_reference = 'controller'
        _BASE_PREPARE(args)
    finally:
        args.max_simulation_sec = simulation_limit
        args.delta_pose_reference = delta_reference
    config = yaml.safe_load(args.engine_config_host.read_text())
    args.expected_runtime_tasks = [
        {'trial': trial_name, 'task_id': task_id,
         **{key: task[key] for key in ('target_module_name', 'port_name', 'plug_name', 'plug_type')}}
        for trial_name, trial in config['trials'].items() for task_id, task in trial['tasks'].items()
    ]
    if len(args.expected_trial_names) != 1 or len(args.expected_runtime_tasks) != 1:
        raise ValueError('A fresh simulator is required for every single Dreamer scene')
    task = args.expected_runtime_tasks[0]
    if (task['target_module_name'], task['port_name']) != ('nic_card_mount_0', 'sfp_port_1'):
        raise ValueError('Strict60 pilot only supports SFP card0/port1')
    source = Path(os.environ['AIC_DREAMER_SOURCE_ROOT']).resolve()
    if not (source / 'dreamer4/aic/models.py').is_file():
        raise ValueError('Missing pinned Dreamer implementation')
    args.dreamer_source_root = str(source)


def runtime_sources(args):
    sources = _BASE_SOURCES(args)
    sources['scripts/evaluate_dreamer_v4_runtime.py'] = Path(__file__).resolve()
    sources['aic_example_policies/aic_example_policies/ros/dreamer_inference_worker.py'] = args.workspace_host/'aic_example_policies/aic_example_policies/ros/dreamer_inference_worker.py'
    sources['dreamer_container_profile.sh'] = args.workspace_host/'outputs/experiments/2026-09-18_dreamer60_pilot/dreamer_container_profile.sh'
    source = Path(args.dreamer_source_root)
    for path in sorted((source / 'dreamer4').rglob('*.py')):
        sources['dreamer_source/' + str(path.relative_to(source))] = path
    return sources


def main(argv=None):
    shared.prepare_args = prepare_args
    shared.runtime_sources = runtime_sources
    return shared.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
