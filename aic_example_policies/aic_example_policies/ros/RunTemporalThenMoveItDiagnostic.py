"""Frozen actor prefix followed by a guarded, privileged MoveIt plan replay.

Development-only Gazebo diagnostic.  The route is an offline simulator-geometry
plan, not an autonomous policy prediction or a successful insertion claim.
"""
import json
import os
from pathlib import Path
import time

import numpy as np
from aic_control_interfaces.msg import JointMotionUpdate, TrajectoryGenerationMode
from trajectory_msgs.msg import JointTrajectoryPoint

from .RunTemporalThenCheatCodeDiagnostic import RunTemporalThenCheatCodeDiagnostic
from .RunTemporalTransport import RunTemporalTransport


NAMES = ('shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
         'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint')


class RunTemporalThenMoveItDiagnostic(RunTemporalThenCheatCodeDiagnostic):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.plan = json.loads(Path(os.environ['AIC_SFP_MOVEIT_PLAN_JSON']).read_text())
        self.variant = os.environ.get('AIC_SFP_MOVEIT_VARIANT', 'retreat_align_approach')
        route = self.plan['route_variants'][self.variant]
        if not route['success'] or len(route['samples']) < 2:
            raise ValueError('Incomplete MoveIt route')
        self.route = route['samples']
        self.audit = Path(os.environ['AIC_TRANSPORT_EVAL_AUDIT'])
        self.live_replan = os.environ.get('AIC_SFP_MOVEIT_LIVE_REPLAN', '0') == '1'

    def _event(self, **values):
        with (self.audit / 'moveit_gazebo_replay.jsonl').open('a') as stream:
            stream.write(json.dumps(values) + '\n')

    @staticmethod
    def _joints(observation):
        by_name = dict(zip(observation.joint_states.name, observation.joint_states.position))
        return np.asarray([by_name[name] for name in NAMES], float)

    @staticmethod
    def _send_joint(move_robot, target):
        move_robot(joint_motion_update=JointMotionUpdate(
            target_state=JointTrajectoryPoint(positions=[float(x) for x in target],
                                              velocities=[0.] * 6),
            target_stiffness=[50., 50., 50., 20., 20., 20.],
            target_damping=[40., 40., 40., 20., 20., 20.],
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION),
            target_feedforward_torque=[0.] * 6))

    def insert_cable(self, task, get_observation, move_robot, send_feedback, **kwargs):
        if (task.target_module_name != self.plan['task']['target_module_name'] or
                task.port_name != self.plan['task']['port_name']):
            raise ValueError('Live task does not match saved MoveIt request')
        self.diagnostic_takeover = False
        RunTemporalTransport.insert_cable(self, task, get_observation, move_robot,
                                          send_feedback, **kwargs)
        if not self.diagnostic_takeover:
            self._event(event='abort', reason='actor_prefix_incomplete')
            return False
        observed = get_observation()
        if observed is None:
            self._event(event='abort', reason='no_takeover_observation')
            return False
        measured_q = self._joints(observed)
        measured_state = self.assembler.assemble_ros(observed)
        measured_tcp = np.asarray(measured_state[:3], float)
        if self.live_replan:
            request = {'schema': 'aic_sfp_moveit_live_start/v1',
                       'task': {'target_module_name': task.target_module_name,
                                'port_name': task.port_name},
                       'sim_s': self._control_sim_time(observed),
                       'joints': measured_q.tolist(),
                       'tcp_base_m': measured_tcp.tolist(),
                       'tcp_quat_xyzw': np.asarray(measured_state[3:7], float).tolist()}
            (self.audit / 'moveit_plan_request.json').write_text(json.dumps(request, indent=2) + '\n')
            deadline = time.monotonic() + 70
            plan_path = self.audit / 'live_moveit_plan.json'
            while not plan_path.exists() and time.monotonic() < deadline:
                if (self.audit / 'live_moveit_plan.error').exists():
                    self._event(event='abort', reason='live_moveit_plan_failed')
                    return False
                time.sleep(.1)
            if not plan_path.exists():
                self._event(event='abort', reason='live_moveit_plan_timeout')
                return False
            self.plan = json.loads(plan_path.read_text())
            self.route = self.plan['route_variants'][self.variant]['samples']
            if not self.plan['route_variants'][self.variant]['success'] or len(self.route) < 2:
                self._event(event='abort', reason='live_moveit_no_route')
                return False
        expected_q = np.asarray(self.route[0]['joints'], float)
        expected_tcp = np.asarray(self.route[0]['tcp_base_m'], float)
        joint_error = float(np.max(np.abs(measured_q - expected_q)))
        tcp_error_mm = float(np.linalg.norm(measured_tcp - expected_tcp) * 1000)
        self._event(event='takeover', sim_s=self._control_sim_time(observed),
                    expected_tcp_base_m=expected_tcp.tolist(), measured_tcp_base_m=measured_tcp.tolist(),
                    joint_max_error_rad=joint_error, tcp_error_mm=tcp_error_mm,
                    requested_card=self.plan['request_card'], requested_port=self.plan['request_port'],
                    variant=self.variant, privileged_plan=True, autonomous_success_claim=False)
        # A route planned from another physical state may drive through a card.
        if joint_error > .05 or tcp_error_mm > 8.:
            self._event(event='abort', reason='takeover_start_mismatch')
            return False
        baseline_force = np.asarray(measured_state[26:29], float)
        high_force_steps = 0
        for i, waypoint in enumerate(self.route[1:], 1):
            self._send_joint(move_robot, waypoint['joints'])
            self.sleep_for(.15)
            current = get_observation()
            if current is None:
                self._event(event='abort', reason='lost_observation', waypoint=i)
                return False
            state = self.assembler.assemble_ros(current)
            q = self._joints(current)
            tcp = np.asarray(state[:3], float)
            command_tcp = np.asarray(waypoint['tcp_base_m'], float)
            force_delta = float(np.linalg.norm(np.asarray(state[26:29]) - baseline_force))
            high_force_steps = high_force_steps + 1 if force_delta > 12. else 0
            self._event(event='waypoint', index=i, sim_s=self._control_sim_time(current),
                        commanded_tcp_base_m=command_tcp.tolist(), measured_tcp_base_m=tcp.tolist(),
                        tracking_error_mm=float(np.linalg.norm(tcp - command_tcp) * 1000),
                        joint_max_error_rad=float(np.max(np.abs(q - np.asarray(waypoint['joints'])))),
                        force_change_n=force_delta)
            if high_force_steps >= 3:
                self._event(event='abort', reason='sustained_force_change_over_12n', waypoint=i)
                return False
        self._send_joint(move_robot, self.route[-1]['joints'])
        self.sleep_for(.7)
        last = get_observation()
        if last is not None:
            state = self.assembler.assemble_ros(last)
            tcp = np.asarray(state[:3], float)
            self._event(event='finished', sim_s=self._control_sim_time(last),
                        measured_tcp_base_m=tcp.tolist(),
                        goal_tcp_error_mm=float(np.linalg.norm(
                            tcp - np.asarray(self.plan['buffer_goal_tcp_base_m'])) * 1000))
        return False
