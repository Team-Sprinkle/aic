"""Frozen visual transport prefix, then privileged CheatCode recovery probe.

This is a same-scene development diagnostic. The actor does not read simulator
geometry; CheatCode and the buffered-arrival test are privileged after takeover.
No result from this policy is an autonomous insertion success.
"""
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from rclpy.time import Time
from tf2_ros import TransformException

from .CheatCode import CheatCode
from .RunTemporalTransport import RunTemporalTransport


class _SafeBufferedArrival(Exception):
    pass


class _TeacherTimeLimit(Exception):
    pass


class _BoundedTransportCheatCode(CheatCode):
    def __init__(self, parent_node, *, audit, duration_s, require_full_insert=False):
        super().__init__(parent_node)
        self.audit = Path(audit)
        self.duration_s = float(duration_s)
        self.require_full_insert = bool(require_full_insert)
        self.start_sim_s = self.time_now().nanoseconds / 1e9
        self.last_geometry = None

    def _send_delta_pose_target(self, move_robot, target_pose):
        if self.time_now().nanoseconds / 1e9 - self.start_sim_s >= self.duration_s:
            raise _TeacherTimeLimit
        super()._send_delta_pose_target(move_robot, target_pose)
        if self._task is None:
            return
        port_frame = f'task_board/{self._task.target_module_name}/{self._task.port_name}_link'
        try:
            port = self._parent_node._tf_buffer.lookup_transform('base_link', port_frame, Time()).transform
            tcp = self._parent_node._tf_buffer.lookup_transform('base_link', 'gripper/tcp', Time()).transform
        except TransformException:
            return
        center = np.array([getattr(port.translation, k) for k in ('x','y','z')], float)
        tcp_xyz = np.array([getattr(tcp.translation, k) for k in ('x','y','z')], float)
        rotation = Rotation.from_quat([getattr(port.rotation, k) for k in ('x','y','z','w')])
        local = rotation.inv().apply(tcp_xyz - center)
        lateral = float(np.linalg.norm(local[:2]))
        self.last_geometry = dict(sim_time_s=self.time_now().nanoseconds / 1e9,
                                  axial_mm=float(local[2]*1000),
                                  lateral_mm=lateral*1000)
        if not self.require_full_insert and -.100 <= local[2] <= -.070 and lateral <= .010:
            (self.audit/'corrective_teacher_buffered_arrival.json').write_text(json.dumps({
                'schema':'aic_privileged_corrective_teacher_buffered_arrival/v1',
                **self.last_geometry,'privileged_geometry_label':True,
                'autonomous_success_claim':False},indent=2)+'\n')
            raise _SafeBufferedArrival


class RunTemporalThenCheatCodeDiagnostic(RunTemporalTransport):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.actor_duration_s = float(os.environ.get('AIC_SFP_CORRECTIVE_ACTOR_DURATION_S','2'))
        self.teacher_duration_s = float(os.environ.get('AIC_SFP_CORRECTIVE_TEACHER_DURATION_S','30'))
        self.require_full_insert = os.environ.get('AIC_SFP_CORRECTIVE_REQUIRE_FULL_INSERT','0')=='1'
        if not 0.5 <= self.actor_duration_s <= 20 or not 1 <= self.teacher_duration_s <= 60:
            raise ValueError('Corrective diagnostic time bounds exceeded')
        self.diagnostic_takeover = False

    def _should_stop_transport(self, task_vector):
        if task_vector[0] != 1:
            raise ValueError('Corrective teacher diagnostic is SFP-only')
        self.diagnostic_takeover = self.last_elapsed_sim_sec >= self.actor_duration_s
        return self.diagnostic_takeover

    def insert_cable(self, task, get_observation, move_robot, send_feedback, **kwargs):
        audit = Path(os.environ['AIC_TRANSPORT_EVAL_AUDIT'])
        self.diagnostic_takeover = False
        super().insert_cable(task, get_observation, move_robot, send_feedback, **kwargs)
        if not self.diagnostic_takeover:
            (audit/'corrective_teacher_takeover.json').write_text(json.dumps({
                'schema':'aic_privileged_corrective_teacher_takeover/v1',
                'actor_prefix_completed':False,'reason':'actor_ended_before_fixed_takeover',
                'autonomous_success_claim':False},indent=2)+'\n')
            return False
        hold = get_observation()
        if hold is not None:
            self.set_delta_pose_target_from_components(
                move_robot=move_robot, delta_position_xyz=np.zeros(3),
                delta_rotation_xyz=np.zeros(3))
        event = dict(schema='aic_privileged_corrective_teacher_takeover/v1',
                     actor_prefix_completed=True,actor_duration_s=self.actor_duration_s,
                     teacher_duration_cap_s=self.teacher_duration_s,
                     takeover_sim_time_s=self.time_now().nanoseconds/1e9,
                     actor_geometry_input=False,privileged_teacher_after_takeover=True,
                     autonomous_success_claim=False)
        (audit/'corrective_teacher_takeover.json').write_text(json.dumps(event,indent=2)+'\n')
        if os.environ.get('AIC_DELAY_TF_LISTENER','0')=='1':
            self._parent_node.enable_tf_listener()
        teacher = _BoundedTransportCheatCode(
            self._parent_node,audit=audit,duration_s=self.teacher_duration_s,
            require_full_insert=self.require_full_insert)
        teacher_success = False
        try:
            teacher_success=bool(teacher.insert_cable(task,get_observation,move_robot,send_feedback))
            outcome=('teacher_returned_full_insert' if teacher_success and self.require_full_insert
                     else 'teacher_returned_without_buffer')
        except _SafeBufferedArrival:
            outcome='safe_buffered_arrival'
        except _TeacherTimeLimit:
            outcome='teacher_time_cap'
        event.update(teacher_outcome=outcome,require_full_insert=self.require_full_insert,
                     last_privileged_geometry=teacher.last_geometry,
                     teacher_end_sim_time_s=self.time_now().nanoseconds/1e9)
        (audit/'corrective_teacher_takeover.json').write_text(json.dumps(event,indent=2)+'\n')
        # The official scorer only records its task end time when the policy
        # reports completion. Keep buffered-arrival diagnostics nonterminal;
        # a full-insertion teacher continuation reports its own result.
        return teacher_success if self.require_full_insert else False
