"""Verify fixed simulation budgets, watchdogs and engine/task log identity without ROS or GPUs."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import uuid

import numpy as np
import pytest
import yaml

HERE = Path(__file__).resolve().parent
SOURCE_ROOT = HERE.parents[1]
REPO = next(p for p in HERE.parents if (p / 'aic_utils/gazebo_rl/gazebo_rl/score_parser.py').is_file())
POLICY = SOURCE_ROOT / 'aic_example_policies/aic_example_policies/ros/RunACTTorchScript.py'
EVALUATOR = SOURCE_ROOT / 'scripts/evaluate_act_checkpoints_runtime.py'
sys.path.insert(0, str(REPO / 'aic_utils/gazebo_rl'))


class Clock:
    now = 0.
    def monotonic(self):
        return self.now
    def sleep(self, duration):
        self.now += duration


def policy_loop(clock):
    tree = ast.parse(POLICY.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'RunACTTorchScript')
    method = deepcopy(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'insert_cable'))
    for argument in method.args.args + method.args.kwonlyargs:
        argument.annotation = None
    method.returns = None
    namespace = {'time': clock, 'np': np, 'json': json, 'uuid': uuid}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(POLICY), 'exec'), namespace)
    return namespace['insert_cable']


def run_loop(*, wall_per_observation, simulation_step=.25, simulation_budget=1., wall_budget=20., missing=False,
             control_clock='simulation'):
    clock, messages = Clock(), []
    node = SimpleNamespace(start_delay_sec=0., control_hz=20., control_clock=control_clock,
        max_runtime_sec=wall_budget, max_simulation_sec=simulation_budget, command_mode='none',
        log_every_n_commands=20, reset_for_task=lambda task: None,
        get_logger=lambda: SimpleNamespace(info=messages.append),
        select_delta_action=lambda observation: np.zeros(6),
        _finite_action_or_none=lambda action, count: action,
        _clamp_action=lambda position, rotation: (position, rotation))
    observations = 0
    def observation():
        nonlocal observations
        clock.now += wall_per_observation
        sim = 123. + observations * simulation_step
        observations += 1
        if missing:
            return None
        stamp = SimpleNamespace(sec=int(sim), nanosec=round((sim % 1) * 1e9))
        return SimpleNamespace(center_image=SimpleNamespace(header=SimpleNamespace(stamp=stamp)))
    task = SimpleNamespace(id='task_1', target_module_name='nic_card_mount_0', port_name='sfp_port_1',
                           plug_name='sfp_tip', plug_type='sfp')
    assert policy_loop(clock)(node, task, observation, lambda *a, **k: None, lambda text: None) is True
    records = [json.loads(line.split('ACT_RUNTIME_STOP ', 1)[1]) for line in messages if line.startswith('ACT_RUNTIME_STOP ')]
    starts = [json.loads(line.split('ACT_RUNTIME_START ', 1)[1]) for line in messages if line.startswith('ACT_RUNTIME_START ')]
    assert len(records) == 1
    assert len(starts) == 1 and all(records[0][key] == value for key, value in starts[0].items())
    return records[0]


@pytest.mark.parametrize('control_clock', ['simulation', 'wall'])
def test_same_simulation_budget_under_different_wall_speeds(control_clock):
    fast = run_loop(wall_per_observation=.1, control_clock=control_clock)
    slow = run_loop(wall_per_observation=.8, control_clock=control_clock)
    for result in [fast, slow]:
        assert result['reason'] == 'simulation_limit'
        assert result['simulation_elapsed_sec'] == 1.
        assert result['commands'] == 4  # Do not command beyond the simulation deadline.
        assert result['simulation_budget_reached'] is True
        assert result['wall_watchdog_shortened'] is False
    assert slow['wall_elapsed_sec'] > 5 * fast['wall_elapsed_sec']


@pytest.mark.parametrize('missing', [False, True])
def test_stalled_simulator_or_missing_observations_hit_watchdog(missing):
    result = run_loop(wall_per_observation=.4, simulation_step=0., wall_budget=1., missing=missing)
    assert result['reason'] == 'wall_watchdog'
    assert result['wall_watchdog_shortened'] is True
    assert result['simulation_budget_reached'] is False
    assert result['simulation_elapsed_sec'] is None if missing else result['simulation_elapsed_sec'] == 0.


def test_disabled_simulation_budget_preserves_wall_stop():
    result = run_loop(wall_per_observation=.4, simulation_budget=None, wall_budget=1.)
    assert result['reason'] == 'wall_watchdog'
    assert result['simulation_limit_sec'] is None
    assert result['wall_watchdog_shortened'] is False


@pytest.fixture
def evaluator():
    spec = importlib.util.spec_from_file_location('proposed_duration_evaluator', EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_logs(tmp_path, records, names):
    policy, engine = tmp_path / 'policy.log', tmp_path / 'engine.log'
    tasks, policy_lines, engine_lines = [], [], []
    keys = ('task_id', 'target_module_name', 'port_name', 'plug_name', 'plug_type')
    for index, (record, name) in enumerate(zip(records, names)):
        identity = {key: record[key] for key in keys}
        tasks.append({'trial': name, **identity})
        start = {**identity, 'runtime_episode_id': record['runtime_episode_id']}
        t = 100. + index * 20
        policy_lines.extend([f'[INFO] [{t+2:.6f}] ACT_RUNTIME_START ' + json.dumps(start),
                             f'[INFO] [{t+2+record["wall_elapsed_sec"]:.6f}] ACT_RUNTIME_STOP ' + json.dumps(record)])
        engine_lines.extend([f'[INFO] [{t:.6f}] Starting trial \'{name}\'',
                             f'[INFO] [{t+1:.6f}] Sending InsertCable goal for task [task_1]'])
    policy.write_text('\n'.join(policy_lines)); engine.write_text('\n'.join(engine_lines))
    return policy, engine, tasks


def test_stop_audit_rejects_watchdog_and_missing_trial(evaluator, tmp_path):
    complete = run_loop(wall_per_observation=.1)
    shortened = run_loop(wall_per_observation=.4, simulation_step=0., wall_budget=1.)
    log, engine, tasks = write_logs(tmp_path, [complete, shortened], ['trial_a', 'trial_b'])
    result = evaluator.audit_simulation_stops(log, engine, ['trial_a', 'trial_b'], tasks, 1.)
    assert result['complete'] is False and result['wall_watchdog_shortened_trials'] == ['trial_b']
    assert not evaluator.evaluation_complete({'simulation_duration_complete': False}, ['trial_a', 'trial_b'])
    log, engine, tasks = write_logs(tmp_path, [complete], ['trial_a'])
    assert evaluator.audit_simulation_stops(log, engine, ['trial_a'], tasks, 1.)['complete'] is True
    assert evaluator.audit_simulation_stops(log, engine, ['trial_a', 'trial_b'], tasks + [tasks[0]], 1.)['complete'] is False
    assert evaluator.audit_simulation_stops(log, engine, ['trial_a'], tasks, 2.)['complete'] is False
    with log.open('a') as file:
        file.write('\nACT_RUNTIME_STOP not valid JSON\n')
    malformed = evaluator.audit_simulation_stops(log, engine, ['trial_a'], tasks, 1.)
    assert malformed['complete'] is False and len(malformed['malformed_stop_records']) == 1


@pytest.mark.parametrize('corruption', ['duplicate_episode', 'wrong_target', 'wrong_engine_order'])
def test_stop_mapping_rejects_duplicate_or_mismatched_tasks(evaluator, tmp_path, corruption):
    records = [run_loop(wall_per_observation=.1), run_loop(wall_per_observation=.2)]
    log, engine, tasks = write_logs(tmp_path, records, ['trial_z', 'trial_a'])
    valid = evaluator.audit_simulation_stops(log, engine, ['trial_z', 'trial_a'], tasks, 1.)
    assert valid['complete'] is True  # task_1 is validly reused across distinct trials.
    if corruption == 'duplicate_episode':
        log.write_text(log.read_text().replace(records[1]['runtime_episode_id'], records[0]['runtime_episode_id']))
    elif corruption == 'wrong_target':
        log.write_text(log.read_text().replace('sfp_port_1', 'sfp_port_0'))
    else:
        engine.write_text(engine.read_text().replace('trial_z', 'TEMP').replace('trial_a', 'trial_z').replace('TEMP', 'trial_a'))
    invalid = evaluator.audit_simulation_stops(log, engine, ['trial_z', 'trial_a'], tasks, 1.)
    assert invalid['complete'] is False and invalid['mapping_errors']


@pytest.mark.parametrize('value', ['nan', 'inf', '0', '-1'])
def test_invalid_simulation_budget_fails_cpu_preflight(evaluator, tmp_path, value):
    args = evaluator.parse_args(['--run-dir', str(tmp_path), '--workspace-host', str(tmp_path),
        '--workspace-container', str(tmp_path), '--max-simulation-sec', value,
        '--policy-module', 'aic_example_policies.ros.RunACTTorchScript', '--command-mode', 'none'])
    with pytest.raises(ValueError, match='Simulation duration must be finite and positive'):
        evaluator.prepare_args(args)


def test_simulation_duration_changes_evaluation_signature(evaluator, tmp_path, monkeypatch):
    checkpoint = tmp_path / 'model.pt'; checkpoint.write_bytes(b'test model identity')
    config = tmp_path / 'scene.yaml'; config.write_text('trials: {}\n')
    monkeypatch.setattr(evaluator, 'runtime_sources', lambda args: {})
    args = SimpleNamespace(act_torchscript=None, engine_config_host=config, max_simulation_sec=60.)
    signature = evaluator.evaluation_signature(checkpoint, args)
    args.max_simulation_sec = 40.
    assert evaluator.evaluation_signature(checkpoint, args) != signature


def test_preflight_preserves_engine_yaml_task_order(evaluator, tmp_path):
    task = {'target_module_name': 'nic_card_mount_0', 'port_name': 'sfp_port_1',
            'plug_name': 'sfp_tip', 'plug_type': 'sfp'}
    config = tmp_path / 'scenes.yaml'
    config.write_text(yaml.safe_dump({'trials': {
        'trial_z': {'tasks': {'task_1': task}},
        'trial_a': {'tasks': {'task_1': {**task, 'port_name': 'sfp_port_0'}}}}}, sort_keys=False))
    model = tmp_path / 'model.pt'; model.write_bytes(b'fixture')
    model.with_suffix('.json').write_text(json.dumps({'checkpoint_dir': str(tmp_path),
        'action_representation': 'absolute_pose', 'chunk_size': 8}))
    (tmp_path / 'policy_preprocessor_step_3_normalizer_processor.safetensors').write_bytes(b'fixture')
    args = evaluator.parse_args(['--run-dir', str(tmp_path), '--workspace-host', str(tmp_path),
        '--workspace-container', str(tmp_path), '--max-simulation-sec', '90', '--max-runtime-sec', '180',
        '--policy-module', 'aic_example_policies.ros.RunACTTorchScript', '--command-mode', 'absolute_pose',
        '--engine-config', str(config), '--act-torchscript', str(model)])
    evaluator.prepare_args(args)
    assert args.expected_trial_order == ['trial_z', 'trial_a']
    assert [(task['trial'], task['task_id'], task['port_name']) for task in args.expected_runtime_tasks] == [
        ('trial_z', 'task_1', 'sfp_port_1'), ('trial_a', 'task_1', 'sfp_port_0')]


@pytest.mark.parametrize('budget, expected_env', [(60., '60.0'), (None, "''")])
def test_evaluator_passes_simulation_budget_to_policy_environment(evaluator, tmp_path, monkeypatch, budget, expected_env):
    args = evaluator.parse_args(['--run-dir', str(tmp_path), '--workspace-host', str(tmp_path),
        '--workspace-container', str(tmp_path), '--max-simulation-sec', '60', '--max-runtime-sec', '180',
        '--command-mode', 'none'])
    args.normalizer_path = None
    args.max_simulation_sec = budget
    args.expected_trial_names = args.expected_trial_order = ['trial_a']
    args.expected_runtime_tasks = []
    args.image_channel_order = 'rgb'
    scripts = []
    monkeypatch.setattr(evaluator, 'runtime_sources', lambda args: {})
    monkeypatch.setattr(evaluator, 'evaluation_signature', lambda *args: 'test signature')
    monkeypatch.setattr(evaluator, 'restart_container', lambda *args: 0)
    monkeypatch.setattr(evaluator.time, 'sleep', lambda seconds: None)
    def capture(args, script, log):
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text('')
        scripts.append(script)
        return object()
    monkeypatch.setattr(evaluator, 'start_long_container_bash', capture)
    monkeypatch.setattr(evaluator, 'wait_for_policy_ready', lambda *args: False)
    monkeypatch.setattr(evaluator, 'stop_process', lambda *args: None)
    checkpoint = tmp_path / 'model.pt'; checkpoint.write_bytes(b'fixture')
    output = tmp_path / 'evaluation'
    result = evaluator.evaluate_checkpoint_once(checkpoint, args, output, output / 'logs', 1)
    assert any('export AIC_ACT_MAX_SIMULATION_SEC=' + expected_env in script for script in scripts)
    assert any('export AIC_ACT_MAX_RUNTIME_SEC=180.0' in script for script in scripts)
    if budget is None:
        assert 'simulation_duration_complete' not in result
    else:
        assert result['simulation_duration_complete'] is False
