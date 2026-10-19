"""Privileged, guarded wide-view capture for SFP identity diagnosis only.

The wide pose is never an inference input or autonomous crop rule. At an
180 mm plug-tip standoff, rise 40 mm, translate +80 mm in base y, hold, then
return before the ordinary CheatCode route resumes. Preserve force/contact
aborts and measured TCP positions for each increment.
"""
import copy
import json
from pathlib import Path

import numpy as np

from .CheatCodeSideViewDiagnostic import CheatCodeSideViewDiagnostic, UnsafeViewMotion


class CheatCodeWideViewDiagnostic(CheatCodeSideViewDiagnostic):
    STANDOFF_M = .180
    ROW_Y_M = .080
    RISE_Z_M = .040

    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        try:
            return super().insert_cable(task, get_observation, move_robot, send_feedback)
        finally:
            Path('/audit/wide_view_events.json').write_text(json.dumps({
                'schema': 'aic_sfp_privileged_wide_view_diagnostic/v1',
                'task_module': task.target_module_name, 'port': task.port_name,
                'standoff_m': self.STANDOFF_M, 'row_y_m': self.ROW_Y_M,
                'rise_z_m': self.RISE_Z_M, 'events': self._events,
                'privileged_motion_only': True,
                'runtime_port_geometry_for_visual_inference': False,
            }, indent=2) + '\n')

    def _approach(self, move_robot, origin, stage, y, z, previous_y, previous_z):
        steps = max(1, int(np.ceil(max(abs(y - previous_y), abs(z - previous_z)) / .002)))
        for j in range(1, steps + 1):
            pose = copy.deepcopy(origin)
            pose.position.y += previous_y + (y - previous_y) * j / steps
            pose.position.z += previous_z + (z - previous_z) * j / steps
            super()._send_delta_pose_target(move_robot, pose)
            self.sleep_for(self.DT_S)
            self._guard(stage, step=j, total=steps,
                        offset_y_m=pose.position.y - origin.position.y,
                        offset_z_m=pose.position.z - origin.position.z)
        for _ in range(20):
            super()._send_delta_pose_target(move_robot, pose)
            self.sleep_for(self.DT_S)
        self._guard(stage + '_hold', requested_offset_y_m=y, requested_offset_z_m=z)
        measured = self._events[-1]['tcp_base_m']
        baseline = self._baseline_tcp
        # The Cartesian controller can lag a fixed one-second hold by a few
        # millimeters. Continue sending the same guarded target briefly; keep
        # the original 5 mm acceptance threshold and all contact limits.
        for j in range(40):
            if abs(measured[1] - baseline[1] - y) <= .005 and abs(measured[2] - baseline[2] - z) <= .005:
                break
            super()._send_delta_pose_target(move_robot, pose)
            self.sleep_for(self.DT_S)
            self._guard(stage + '_settling', step=j + 1,
                        requested_offset_y_m=y, requested_offset_z_m=z)
            measured = self._events[-1]['tcp_base_m']
        if abs(measured[1] - baseline[1] - y) > .005 or abs(measured[2] - baseline[2] - z) > .005:
            raise UnsafeViewMotion(f'{stage}: measured TCP offset differs by >5 mm')

    def _view_sweep(self, move_robot, origin):
        self._guard('baseline_before_hold')
        for _ in range(20):
            super()._send_delta_pose_target(move_robot, origin)
            self.sleep_for(self.DT_S)
        self._guard('baseline_view')
        self._baseline_tcp = self._events[-1]['tcp_base_m']
        self._approach(move_robot, origin, 'rising', 0., self.RISE_Z_M, 0., 0.)
        self._approach(move_robot, origin, 'wide_view', self.ROW_Y_M,
                       self.RISE_Z_M, 0., self.RISE_Z_M)
        self._approach(move_robot, origin, 'return_y', 0., self.RISE_Z_M,
                       self.ROW_Y_M, self.RISE_Z_M)
        self._approach(move_robot, origin, 'return_z', 0., 0., 0., self.RISE_Z_M)
