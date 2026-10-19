"""Privileged short teacher prefix, then frozen temporal BC, for diagnosis only.

Never use this policy for autonomous scores or deployment. The prefix is an
offline controller-reference trace from a successful run on the *same scene*.
"""
import json
import os
from pathlib import Path

import numpy as np

from .RunTemporalTransport import RunTemporalTransport
from shared_port_tcp_geometry import matrix_from_xyz_xyzw


class RunTemporalTeacherPrefixDiagnostic(RunTemporalTransport):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        path = Path(os.environ['AIC_TEMPORAL_TEACHER_PREFIX_FILE'])
        data = json.loads(path.read_text())
        if data['schema'] != 'aic_sfp_privileged_teacher_prefix/v1' or not data['diagnostic_only']:
            raise ValueError('Expected explicitly privileged diagnostic prefix')
        self.prefix_duration_s = float(data['duration_s'])
        self.prefix_teacher_start_s = float(data['teacher_start_sim_s'])
        poses = data['reference_base_poses']
        self.prefix_times = np.asarray([r['sim_time_s'] for r in poses], float)
        self.prefix_base_poses = np.asarray([r['pose_xyz_xyzw'] for r in poses], float)
        if len(poses) < 50 or self.prefix_duration_s <= 0 or self.prefix_duration_s > 8:
            raise ValueError('Invalid bounded teacher prefix')
        self.prefix_takeover_logged = False

    def predict_chunk(self, observation, task_vector, *, decision_time_s=None):
        targets, estimated_pose, state = super().predict_chunk(
            observation, task_vector, decision_time_s=decision_time_s)
        if task_vector[0] != 1:
            raise ValueError('Teacher prefix diagnostic is SFP-only')
        stamp = observation.center_image.header.stamp
        sim_time = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        elapsed = sim_time - self.first_sim_time
        if elapsed >= self.prefix_duration_s:
            if not self.prefix_takeover_logged:
                self._trace(event='diagnostic_teacher_prefix_takeover',
                            sim_time=sim_time,elapsed_s=elapsed,
                            privileged_input_ended=True)
                self.prefix_takeover_logged = True
            return targets, estimated_pose, state
        # The same-scene teacher clock is aligned to the first actor decision.
        # Each reference is a full target in base_link, converted into a full
        # TCP-body target using the *measured* current TCP as anchor.
        current = matrix_from_xyz_xyzw(state[:7])
        inverse = np.linalg.inv(current)
        body_targets = []
        selected_times = []
        for step in range(len(targets)):
            teacher_time = self.prefix_teacher_start_s + elapsed + step / self.control_hz
            index = int(np.argmin(np.abs(self.prefix_times - teacher_time)))
            selected_times.append(float(self.prefix_times[index]))
            base_target = matrix_from_xyz_xyzw(self.prefix_base_poses[index])
            body = inverse @ base_target
            body_targets.append(np.r_[body[:3,3],body[:3,0],body[:3,1]])
        body_targets = np.asarray(body_targets, np.float32)
        self.latest_proposal_chunk = body_targets.copy()
        self._trace(event='diagnostic_teacher_prefix_proposal',sim_time=sim_time,
                    elapsed_s=elapsed,teacher_reference_times_s=selected_times,
                    privileged_input=True,targets=body_targets.tolist())
        return body_targets, estimated_pose, state
