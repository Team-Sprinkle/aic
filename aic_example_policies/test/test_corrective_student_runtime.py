"""The corrective collector must preserve the student's runtime state contract."""
import ast
import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from test_act_task_contract import runtime_methods

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'aic_utils/lerobot_robot_aic'))
from lerobot_robot_aic.runtime_features import AICRuntimeFeatureAssembler
from lerobot_robot_aic.task_encoding import encode_task_vector


def collector_methods(*names):
    source = ROOT / 'aic_example_policies/aic_example_policies/ros/CollectCorrectiveCheatCode.py'
    cls = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef))
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for method in methods:
        method.decorator_list = []
    scope = {'np': np, 'math': math, 'copy': copy}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), 'exec'), scope)
    return scope


@pytest.mark.parametrize('base_dim', [32, 42])
def test_student_uses_runtime_task_features_and_bounds_actual_intervention(base_dim):
    methods = runtime_methods('_task_vector', '_parse_index_from_suffix', '_state_vector')
    task = NS(target_module_name='nic_card_mount_0', port_name='sfp_port_1')
    assembler = AICRuntimeFeatureAssembler(base_dim)
    base_state = np.arange(32, dtype=np.float32)
    assembler.assemble_ros = lambda observation: assembler.assemble(base_state)
    student = NS(_current_task=task, feature_assembler=assembler,
                 _parse_index_from_suffix=methods['_parse_index_from_suffix'],
                 include_elapsed_sim_time=True, _episode_first_image_time=None,
                 time_clip_sec=40., quaternion_sign='x')
    student._task_vector = lambda task: methods['_task_vector'](student, task)
    observed_states = []

    def predict(observation):
        observed_states.append(methods['_state_vector'](student, observation))
        return np.array([.1, 0., 0., .2, 0., 0.])

    student.select_delta_action = predict
    observation = NS(center_image=NS(header=NS(stamp=NS(sec=100, nanosec=0))))
    collector = NS(observe=lambda: observation, student=student, first_time=None,
                   last_recorded_time=100., command_count=90, cycle=1,
                   offset=np.zeros(5), perturbation_scale=0., student_cycle=1,
                   student_cycle_selected=True, student_probability=.25)
    funcs = collector_methods('_send_delta_pose_target', '_pose_values')
    collector._pose_values = funcs['_pose_values']
    collector._execute_target = lambda callback, pose: pose
    teacher = NS(position=NS(x=0., y=0., z=0.), orientation=NS(x=0., y=0., z=0., w=1.))
    executed = funcs['_send_delta_pose_target'](collector, None, teacher)

    assert observed_states[0].shape == (base_dim + 1,)
    np.testing.assert_array_equal(observed_states[0][:32], base_state)
    if base_dim == 42:
        canonical = encode_task_vector(task_family='sfp_to_nic', target_port_index=1, target_card_index=0)
        np.testing.assert_array_equal(observed_states[0][32:42], canonical)
    assert observed_states[0][-1] == 0.
    position = np.array([executed.position.x, executed.position.y, executed.position.z])
    rotation = Rotation.from_quat([executed.orientation.x, executed.orientation.y,
                                  executed.orientation.z, executed.orientation.w])
    assert np.linalg.norm(position) == pytest.approx(.03)
    assert rotation.magnitude() == pytest.approx(.08)
    assert teacher.position.x == 0. and teacher.orientation.w == 1.
