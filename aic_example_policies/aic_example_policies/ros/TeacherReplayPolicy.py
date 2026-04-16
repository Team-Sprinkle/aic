#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#

from __future__ import annotations

import json
import os

from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model.policy import GetObservationCallback, MoveRobotCallback, Policy, SendFeedbackCallback
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Twist, Vector3, Wrench


class TeacherReplayPolicy(Policy):
    """Replay a saved teacher trajectory through the official `aic_model` path."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._artifact_path = os.environ.get("AIC_TEACHER_REPLAY_ARTIFACT")
        if not self._artifact_path:
            raise RuntimeError(
                "TeacherReplayPolicy requires AIC_TEACHER_REPLAY_ARTIFACT to be set."
            )
        rank_env = os.environ.get("AIC_TEACHER_REPLAY_CANDIDATE_RANK")
        self._candidate_rank = int(rank_env) if rank_env else None

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        del task, get_observation
        from aic_gym_gz.teacher.official_replay import load_official_replay_sequence

        artifact = load_official_replay_sequence(
            self._artifact_path,
            candidate_rank=self._candidate_rank,
        )
        send_feedback(f"Loaded teacher replay artifact from {self._artifact_path}")
        for segment_index, segment in enumerate(artifact.get("trajectory_segments", [])):
            send_feedback(
                f"Replaying segment {segment_index + 1}/{len(artifact.get('trajectory_segments', []))}"
            )
            for point in segment.get("points", []):
                action = point["action"]
                motion_update = MotionUpdate()
                motion_update.header.stamp = self.time_now().to_msg()
                motion_update.header.frame_id = "base_link"
                motion_update.velocity = Twist(
                    linear=Vector3(
                        x=float(action[0]),
                        y=float(action[1]),
                        z=float(action[2]),
                    ),
                    angular=Vector3(
                        x=float(action[3]),
                        y=float(action[4]),
                        z=float(action[5]),
                    ),
                )
                motion_update.target_stiffness = [
                    85.0 if (i % 7 == 0) else 0.0 for i in range(36)
                ]
                motion_update.target_damping = [
                    75.0 if (i % 7 == 0) else 0.0 for i in range(36)
                ]
                motion_update.feedforward_wrench_at_tip = Wrench(
                    force=Vector3(x=0.0, y=0.0, z=0.0),
                    torque=Vector3(x=0.0, y=0.0, z=0.0),
                )
                motion_update.wrench_feedback_gains_at_tip = [0.0] * 6
                motion_update.trajectory_generation_mode.mode = (
                    TrajectoryGenerationMode.MODE_VELOCITY
                )
                move_robot(motion_update=motion_update)
                self.sleep_for(float(point.get("dt", 0.02)))
        hold = MotionUpdate()
        hold.header.stamp = self.time_now().to_msg()
        hold.header.frame_id = "base_link"
        hold.velocity = Twist(
            linear=Vector3(x=0.0, y=0.0, z=0.0),
            angular=Vector3(x=0.0, y=0.0, z=0.0),
        )
        hold.target_stiffness = [85.0 if (i % 7 == 0) else 0.0 for i in range(36)]
        hold.target_damping = [75.0 if (i % 7 == 0) else 0.0 for i in range(36)]
        hold.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        hold.wrench_feedback_gains_at_tip = [0.0] * 6
        hold.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY
        move_robot(motion_update=hold)
        send_feedback("Teacher replay finished.")
        return True
