"""Guarded local-x wrist view at 180 mm SFP standoff, diagnostic only.

The selected port TF controls where the privileged teacher pauses, but no
geometry enters visual review or a deployable actor. Abort on inherited force
and oracle contact rules and require measured position/orientation return.
"""

import copy
import json
from pathlib import Path

import numpy as np
from rclpy.time import Time
from scipy.spatial.transform import Rotation

from .CheatCodeSideViewDiagnostic import CheatCodeSideViewDiagnostic, UnsafeViewMotion


class CheatCodeBoardTiltViewDiagnostic(CheatCodeSideViewDiagnostic):
    STANDOFF_M = .180
    TILT_DEG = 10.0
    STEP_DEG = 1.0

    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        try:
            return super().insert_cable(task, get_observation, move_robot, send_feedback)
        finally:
            Path('/audit/board_tilt_view_events.json').write_text(json.dumps({
                'schema': 'aic_sfp_privileged_board_tilt_view/v1',
                'task_module': task.target_module_name, 'port': task.port_name,
                'standoff_m': self.STANDOFF_M, 'tilt_local_x_deg': self.TILT_DEG,
                'events': self._events, 'privileged_motion_only': True,
                'runtime_port_geometry_for_visual_inference': False,
            }, indent=2) + '\n')

    def _measured_tcp(self):
        tf = self._parent_node._tf_buffer.lookup_transform('base_link', 'gripper/tcp', Time())
        p, q = tf.transform.translation, tf.transform.rotation
        return np.array([p.x, p.y, p.z]), Rotation.from_quat([q.x, q.y, q.z, q.w])

    def _target(self, origin, angle_deg):
        pose = copy.deepcopy(origin)
        q = origin.orientation
        rot = Rotation.from_quat([q.x, q.y, q.z, q.w]) * Rotation.from_euler('x', angle_deg, degrees=True)
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = (
            map(float, rot.as_quat()))
        return pose

    def _hold_and_check(self, move_robot, origin, angle_deg, stage, base_position, base_rotation):
        target = self._target(origin, angle_deg)
        for step in range(40):
            super()._send_delta_pose_target(move_robot, target)
            self.sleep_for(self.DT_S)
            self._guard(stage + '_settling', step=step + 1, target_local_x_deg=angle_deg)
            pos, rot = self._measured_tcp()
            expected = base_rotation * Rotation.from_euler('x', angle_deg, degrees=True)
            error = (expected.inv() * rot).magnitude() * 180 / np.pi
            if np.linalg.norm((pos - base_position) * 1000) <= 5 and error <= 2:
                break
        pos, rot = self._measured_tcp()
        expected = base_rotation * Rotation.from_euler('x', angle_deg, degrees=True)
        error = (expected.inv() * rot).magnitude() * 180 / np.pi
        pos_error = np.linalg.norm((pos - base_position) * 1000)
        self._guard(stage + '_hold', measured_position_error_mm=float(pos_error),
                    measured_orientation_error_deg=float(error), target_local_x_deg=angle_deg)
        if pos_error > 5 or error > 2:
            raise UnsafeViewMotion(f'{stage}: measured pose error {pos_error:.2f} mm/{error:.2f} deg')

    def _view_sweep(self, move_robot, origin):
        self._guard('baseline_before_hold')
        for _ in range(20):
            super()._send_delta_pose_target(move_robot, origin)
            self.sleep_for(self.DT_S)
        self._guard('baseline_view')
        base_position, base_rotation = self._measured_tcp()
        # The teacher's next absolute orientation can already be tens of
        # degrees from the measured held wrist pose. A diagnostic *relative*
        # view must start from the measured pose, not that future target.
        held_pose = copy.deepcopy(origin)
        held_pose.position.x, held_pose.position.y, held_pose.position.z = map(float, base_position)
        held_pose.orientation.x, held_pose.orientation.y, held_pose.orientation.z, held_pose.orientation.w = (
            map(float, base_rotation.as_quat()))
        n = max(1, int(np.ceil(abs(self.TILT_DEG) / self.STEP_DEG)))
        for j in range(1, n + 1):
            angle = self.TILT_DEG * j / n
            super()._send_delta_pose_target(move_robot, self._target(held_pose, angle))
            self.sleep_for(self.DT_S)
            self._guard('tilting', step=j, total=n, target_local_x_deg=angle)
        self._hold_and_check(move_robot, held_pose, self.TILT_DEG, 'tilted_view', base_position, base_rotation)
        for j in range(1, n + 1):
            angle = self.TILT_DEG * (1 - j / n)
            super()._send_delta_pose_target(move_robot, self._target(held_pose, angle))
            self.sleep_for(self.DT_S)
            self._guard('returning', step=j, total=n, target_local_x_deg=angle)
        self._hold_and_check(move_robot, held_pose, 0.0, 'return', base_position, base_rotation)
