"""Causal estimated-frame supervision must preserve the executed full target."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from build_verified_predicted_port_actor_rows import convert_record, pose9
from shared_port_tcp_geometry import matrix_from_pose9


def fixture():
    port=np.eye(4);port[:3,:3]=Rotation.from_euler('xyz',[.1,-.2,.8]).as_matrix()
    port[:3,3]=[.3,-.4,.2]
    tcp=np.eye(4);tcp[:3,:3]=Rotation.from_euler('xyz',[.2,.3,-.1]).as_matrix()
    tcp[:3,3]=[.25,-.3,.4]
    target=tcp.copy();target[:3,3]+=[.03,-.02,.01]
    task=[0.,1.,1.,0.,0.,0.,0.,0.,0.,0.]
    row=dict(family='sc',episode='e',scene='reset',split='train',task=task,frame=0,
        time_s=2.,images={},image_kind='native_1152x1024_rgb_jpeg',
        state=[*tcp[:3,3],*Rotation.from_matrix(tcp[:3,:3]).as_quat(),*([0.]*25)],
        source_stream='original',source_command_indices=[1,1,2,2],
        action_target_offsets_s=[.05,.10,.15,.20],
        recorded_controller_stiffness=[90,90,90,50,50,50],
        recorded_controller_damping=[50,50,50,20,20,20],
        recorded_controller_wrench_feedback_gains=[.5,.5,.5,0.,0.,0.],
        recorded_feedforward_wrench=[0.]*6,recorded_trajectory_mode=2,
        full_recorded_controller_contract_verified=True,
        body_command_conversion='audited approximation',action_owner='teacher',
        post_completion_commands_excluded=True,
        labels_are_recorded_targets_not_measured_future_motion=True,
        recorded_base_tcp_targets_pose9=[pose9(target)]*4)
    lock=dict(episode='e',scene='reset',split='train',task=task,time_s=1.5,
        base_port_matrix=port.tolist(),geometry_input=False)
    return row,lock,target


def test_complete_command_survives_estimated_frame_roundtrip():
    row,lock,target=fixture();original=copy.deepcopy(row)
    result=convert_record(row,lock)
    actual=np.asarray(lock['base_port_matrix'])@matrix_from_pose9(
        result['teacher_tcp_targets_predicted_port_pose9'][0])
    np.testing.assert_allclose(actual,target,atol=1e-12)
    assert row==original
    assert 'supervision_only_TCP_axial_m' not in result


def test_future_lock_is_not_backfilled():
    row,lock,_=fixture();lock['time_s']=row['time_s']+.01
    assert convert_record(row,lock) is None


@pytest.mark.parametrize('field,value',[
    ('scene','another_reset'),('task',[1.]*10),('geometry_input',True)])
def test_reset_query_and_privileged_lock_rejected(field,value):
    row,lock,_=fixture();lock[field]=value
    with pytest.raises(ValueError):convert_record(row,lock)


@pytest.mark.parametrize('field,value',[
    ('action_owner','dataset_reset_supervisor'),
    ('post_completion_commands_excluded',False),
    ('labels_are_recorded_targets_not_measured_future_motion',False),
    ('recorded_controller_stiffness',[240,240,240,50,50,50])])
def test_invalid_label_or_controller_contract_rejected(field,value):
    row,lock,_=fixture();row[field]=value
    with pytest.raises(ValueError):convert_record(row,lock)


@pytest.mark.parametrize('field,value',[
    ('recorded_controller_wrench_feedback_gains',[0.]*6),
    ('recorded_feedforward_wrench',[1.,0.,0.,0.,0.,0.]),
    ('recorded_trajectory_mode',1)])
def test_mismatched_force_or_trajectory_mode_rejected(field,value):
    row,lock,_=fixture();row[field]=value
    with pytest.raises(ValueError):convert_record(row,lock)
