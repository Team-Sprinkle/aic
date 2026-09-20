"""The Dreamer policy-name exception preserves shared ACT and audit gates."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import evaluate_dreamer_v4_runtime as evaluator


def settings(tmp_path):
    config=tmp_path/'scene.yaml'
    config.write_text(yaml.safe_dump({'trials':{'one':{'tasks':{'insert':{'target_module_name':'nic_card_mount_0','port_name':'sfp_port_1','plug_name':'sfp_cable','plug_type':'sfp'}}}}}))
    return SimpleNamespace(policy_module=evaluator.POLICY,diagnostic_ground_truth=False,corrective_data_dir=None,act_torchscript=None,max_simulation_sec=90.,max_runtime_sec=180.,command_mode='delta_pose',command_frame='gripper/tcp',delta_pose_reference='observation',control_hz=20.,control_clock='simulation',n_action_steps=4,temporal_ensemble_coeff=None,start_delay_sec=0,max_translation_delta=.1,max_rotation_delta=.2,translation_limit_mode='norm',translation_deadband=0.,rotation_deadband=0.,engine_config_host=config,expected_trial_names=['one'])


def test_only_specific_dreamer_policy_allowed(tmp_path):
    a=settings(tmp_path);a.policy_module='aic_example_policies.ros.OtherPolicy'
    with patch.object(evaluator,'_BASE_PREPARE') as common,pytest.raises(ValueError,match='exclusively'):
        evaluator.prepare_args(a)
    common.assert_not_called()


def test_full_duration_and_task_identity_restored(tmp_path):
    a=settings(tmp_path);source=tmp_path/'source';(source/'dreamer4/aic').mkdir(parents=True);(source/'dreamer4/aic/models.py').touch()
    def common(args):
        assert args.max_simulation_sec is None
        assert args.delta_pose_reference == 'controller'
        args.expected_runtime_tasks=[]
    with patch.object(evaluator,'_BASE_PREPARE',side_effect=common),patch.dict(os.environ,{'AIC_DREAMER_SOURCE_ROOT':str(source)}):evaluator.prepare_args(a)
    assert a.max_simulation_sec==90.
    assert a.delta_pose_reference=='observation'
    assert a.expected_runtime_tasks==[{'trial':'one','task_id':'insert','target_module_name':'nic_card_mount_0','port_name':'sfp_port_1','plug_name':'sfp_cable','plug_type':'sfp'}]


@pytest.mark.parametrize('key,value',[('max_simulation_sec',60.),('max_runtime_sec',90.),('diagnostic_ground_truth',True),('n_action_steps',1),('translation_deadband',.001),('delta_pose_reference','controller'),('command_mode','absolute_pose')])
def test_protocol_changes_rejected(tmp_path,key,value):
    a=settings(tmp_path);setattr(a,key,value)
    with patch.object(evaluator,'_BASE_PREPARE') as common,pytest.raises(ValueError):evaluator.prepare_args(a)
    common.assert_not_called()


def test_real_shared_preflight_accepts_only_exact_inherited_delta_contract(tmp_path):
    template=settings(tmp_path)
    source=tmp_path/'source';(source/'dreamer4/aic').mkdir(parents=True)
    (source/'dreamer4/aic/models.py').touch()
    args=evaluator.shared.parse_args([
        '--run-dir',str(tmp_path),'--workspace-host',str(tmp_path),
        '--engine-config',str(template.engine_config_host),'--policy-module',evaluator.POLICY,
        '--command-mode','delta_pose','--command-frame','gripper/tcp',
        '--delta-pose-reference','observation','--control-clock','simulation',
        '--max-runtime-sec','180','--max-simulation-sec','90',
        '--max-translation-delta','.1','--max-rotation-delta','.2',
        '--translation-limit-mode','norm','--translation-deadband','0',
        '--rotation-deadband','0','--n-action-steps','4'])
    with patch.dict(os.environ,{'AIC_DREAMER_SOURCE_ROOT':str(source)}):
        evaluator.prepare_args(args)
    assert args.delta_pose_reference=='observation' and args.max_simulation_sec==90.
    assert len(args.expected_runtime_tasks)==1


def test_common_failure_restores_observed_delta_and_duration(tmp_path):
    args=settings(tmp_path)
    with patch.object(evaluator,'_BASE_PREPARE',side_effect=ValueError('test failure')):
        with pytest.raises(ValueError,match='test failure'):
            evaluator.prepare_args(args)
    assert args.delta_pose_reference=='observation' and args.max_simulation_sec==90.
