#!/usr/bin/env python3

#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""Record autonomous policy rollouts into native LeRobot dataset format.

This recorder listens to:
- `/observations` from `aic_model`
- `/aic_controller/pose_commands` and `/aic_controller/joint_commands`
- `/insert_cable/_action/status`

and writes episodes as a standard LeRobot dataset (same schema family as
`lerobot-record`) so it can be trained with `lerobot-train` and merged via
`--resume` into existing teleoperation datasets.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from aic_control_interfaces.msg import JointMotionUpdate, MotionUpdate, TrajectoryGenerationMode
from aic_engine_interfaces.srv import GetEpisodeSaveStatus
from aic_model_interfaces.msg import Observation
from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.processor import make_default_processors
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import sanity_check_dataset_robot_compatibility
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from aic_model.policy import quaternion_xyzw_to_rotation_vector
from .types import JointMotionUpdateActionDict, MotionUpdateActionDict


ACTIVE_GOAL_STATES = {
    GoalStatus.STATUS_ACCEPTED,
    GoalStatus.STATUS_EXECUTING,
    GoalStatus.STATUS_CANCELING,
}
TERMINAL_GOAL_STATES = {
    GoalStatus.STATUS_SUCCEEDED,
    GoalStatus.STATUS_CANCELED,
    GoalStatus.STATUS_ABORTED,
}


@dataclass
class _EpisodeSaveState:
    finalized: bool = False
    saved: bool = False
    terminal_state: int = 0
    message: str = "Waiting for terminal action status."


def _fixed_len(values: list[float], length: int) -> list[float]:
    out = list(values[:length])
    if len(out) < length:
        out.extend([0.0] * (length - len(out)))
    return out


def _stamp_to_sec(stamp: Any) -> float | None:
    sec = int(getattr(stamp, "sec", 0))
    nanosec = int(getattr(stamp, "nanosec", 0))
    if sec == 0 and nanosec == 0:
        return None
    return float(sec) + float(nanosec) * 1e-9


class _RobotTypeShim:
    def __init__(self, robot_type: str):
        self.robot_type = robot_type


class PolicyRecorder(Node):
    def __init__(
        self,
        repo_id: str,
        single_task: str,
        root: Path | None,
        fps: int,
        video: bool,
        vcodec: str,
        image_writer_processes: int,
        image_writer_threads_per_camera: int,
        video_encoding_batch_size: int,
        resume: bool,
        push_to_hub: bool,
        private: bool,
        tags: list[str] | None,
        save_failed_episodes: bool,
        max_episodes: int,
        action_mode: str,
        action_timeout_sec: float,
        camera_scale: float,
        observations_topic: str,
        pose_commands_topic: str,
        joint_commands_topic: str,
        status_topic: str,
        status_service_name: str,
        robot_type: str,
    ):
        super().__init__("aic_policy_recorder")

        self.repo_id = repo_id
        self.single_task = single_task
        self.root = root
        self.fps = fps
        self.video = video
        self.vcodec = vcodec
        self.image_writer_processes = image_writer_processes
        self.image_writer_threads_per_camera = image_writer_threads_per_camera
        self.video_encoding_batch_size = video_encoding_batch_size
        self.resume = resume
        self.push_to_hub = push_to_hub
        self.private = private
        self.tags = tags
        self.save_failed_episodes = save_failed_episodes
        self.max_episodes = max_episodes
        self.action_mode = action_mode
        self.action_timeout_sec = action_timeout_sec
        self.camera_scale = camera_scale
        self.robot_type = robot_type

        self._dataset: LeRobotDataset | None = None
        self._dataset_ready = False
        self._recording = False
        self._current_goal_id: tuple[int, ...] | None = None
        self._pending_terminal_state: int | None = None
        self._episodes_saved = 0
        self._unsupported_encodings_logged: set[str] = set()
        self._unsupported_action_modes_logged: set[str] = set()
        self._episode_save_state_by_goal: dict[tuple[int, ...], _EpisodeSaveState] = {}

        self._latest_obs: Observation | None = None
        self._latest_motion_cmd: MotionUpdate | None = None
        self._latest_joint_cmd: JointMotionUpdate | None = None
        self._latest_motion_recv_time = 0.0
        self._latest_joint_wall_time = 0.0

        self._teleop_action_processor, _, self._robot_observation_processor = (
            make_default_processors()
        )

        self.create_subscription(
            Observation,
            observations_topic,
            self._observation_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            MotionUpdate,
            pose_commands_topic,
            self._motion_cmd_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            JointMotionUpdate,
            joint_commands_topic,
            self._joint_cmd_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            GoalStatusArray,
            status_topic,
            self._status_cb,
            qos_profile_sensor_data,
        )
        self.create_service(
            GetEpisodeSaveStatus,
            status_service_name,
            self._get_episode_save_status_cb,
        )

        self.create_timer(1.0 / float(self.fps), self._sample_frame)

        self.get_logger().info(
            "Policy recorder started. Waiting for first observation to initialize dataset..."
        )
        self.get_logger().info(
            f"Episode save status service ready at '{status_service_name}'."
        )

    def _set_goal_save_state(
        self,
        goal_id: tuple[int, ...],
        *,
        finalized: bool,
        saved: bool,
        terminal_state: int,
        message: str,
    ) -> None:
        self._episode_save_state_by_goal[goal_id] = _EpisodeSaveState(
            finalized=finalized,
            saved=saved,
            terminal_state=terminal_state,
            message=message,
        )

    def _get_episode_save_status_cb(
        self,
        request: GetEpisodeSaveStatus.Request,
        response: GetEpisodeSaveStatus.Response,
    ) -> GetEpisodeSaveStatus.Response:
        goal_id = tuple(int(x) for x in request.goal_id)
        state = self._episode_save_state_by_goal.get(goal_id)
        if state is None:
            response.goal_known = False
            response.finalized = False
            response.saved = False
            response.terminal_state = 0
            response.message = f"Goal {goal_id} is unknown to recorder."
            return response

        response.goal_known = True
        response.finalized = state.finalized
        response.saved = state.saved
        response.terminal_state = int(state.terminal_state)
        response.message = state.message
        return response

    def _status_cb(self, msg: GoalStatusArray) -> None:
        status_by_goal: dict[tuple[int, ...], int] = {}
        for status in msg.status_list:
            goal_id = tuple(int(x) for x in status.goal_info.goal_id.uuid)
            status_by_goal[goal_id] = int(status.status)

        if self._current_goal_id is None:
            for goal_id, state in status_by_goal.items():
                if state in ACTIVE_GOAL_STATES:
                    self._recording = True
                    self._current_goal_id = goal_id
                    self._pending_terminal_state = None
                    self._set_goal_save_state(
                        goal_id,
                        finalized=False,
                        saved=False,
                        terminal_state=0,
                        message="Goal active. Waiting for terminal action status.",
                    )
                    self.get_logger().info(
                        f"Episode started from goal {goal_id}."
                    )
                    return
            return

        if self._current_goal_id not in status_by_goal:
            return

        state = status_by_goal[self._current_goal_id]
        if state in TERMINAL_GOAL_STATES:
            self._pending_terminal_state = state
            self._set_goal_save_state(
                self._current_goal_id,
                finalized=False,
                saved=False,
                terminal_state=state,
                message=(
                    f"Terminal action status {state} observed. "
                    "Waiting for episode save finalization."
                ),
            )

    def _observation_cb(self, msg: Observation) -> None:
        self._latest_obs = msg

    def _motion_cmd_cb(self, msg: MotionUpdate) -> None:
        self._latest_motion_cmd = msg
        self._latest_motion_recv_time = time.monotonic()

    def _joint_cmd_cb(self, msg: JointMotionUpdate) -> None:
        self._latest_joint_cmd = msg
        self._latest_joint_wall_time = time.time()

    def _image_to_bgr(self, image_msg: Image) -> np.ndarray:
        h = int(image_msg.height)
        w = int(image_msg.width)
        step = int(image_msg.step)
        if h <= 0 or w <= 0:
            return np.zeros((1, 1, 3), dtype=np.uint8)

        encoding = image_msg.encoding.lower()
        row_data = np.frombuffer(image_msg.data, dtype=np.uint8)
        try:
            row_data = row_data.reshape(h, step)
        except ValueError:
            return np.zeros((h, w, 3), dtype=np.uint8)

        if encoding == "bgr8":
            image = row_data[:, : w * 3].reshape(h, w, 3)
        elif encoding == "rgb8":
            rgb = row_data[:, : w * 3].reshape(h, w, 3)
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif encoding == "mono8":
            gray = row_data[:, :w]
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif encoding == "bgra8":
            bgra = row_data[:, : w * 4].reshape(h, w, 4)
            image = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        elif encoding == "rgba8":
            rgba = row_data[:, : w * 4].reshape(h, w, 4)
            image = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        else:
            if encoding not in self._unsupported_encodings_logged:
                self.get_logger().warn(
                    f"Unsupported image encoding '{image_msg.encoding}'. Using blank image for this stream."
                )
                self._unsupported_encodings_logged.add(encoding)
            image = np.zeros((h, w, 3), dtype=np.uint8)

        if self.camera_scale != 1.0:
            image = cv2.resize(
                image,
                None,
                fx=self.camera_scale,
                fy=self.camera_scale,
                interpolation=cv2.INTER_AREA,
            )
        return image

    def _obs_message_to_values(self, obs: Observation) -> dict[str, Any]:
        ctrl = obs.controller_state

        values: dict[str, Any] = {
            "tcp_pose.position.x": float(ctrl.tcp_pose.position.x),
            "tcp_pose.position.y": float(ctrl.tcp_pose.position.y),
            "tcp_pose.position.z": float(ctrl.tcp_pose.position.z),
            "tcp_pose.orientation.x": float(ctrl.tcp_pose.orientation.x),
            "tcp_pose.orientation.y": float(ctrl.tcp_pose.orientation.y),
            "tcp_pose.orientation.z": float(ctrl.tcp_pose.orientation.z),
            "tcp_pose.orientation.w": float(ctrl.tcp_pose.orientation.w),
            "tcp_velocity.linear.x": float(ctrl.tcp_velocity.linear.x),
            "tcp_velocity.linear.y": float(ctrl.tcp_velocity.linear.y),
            "tcp_velocity.linear.z": float(ctrl.tcp_velocity.linear.z),
            "tcp_velocity.angular.x": float(ctrl.tcp_velocity.angular.x),
            "tcp_velocity.angular.y": float(ctrl.tcp_velocity.angular.y),
            "tcp_velocity.angular.z": float(ctrl.tcp_velocity.angular.z),
            "tcp_error.x": float(ctrl.tcp_error[0]),
            "tcp_error.y": float(ctrl.tcp_error[1]),
            "tcp_error.z": float(ctrl.tcp_error[2]),
            "tcp_error.rx": float(ctrl.tcp_error[3]),
            "tcp_error.ry": float(ctrl.tcp_error[4]),
            "tcp_error.rz": float(ctrl.tcp_error[5]),
            "joint_positions.0": 0.0,
            "joint_positions.1": 0.0,
            "joint_positions.2": 0.0,
            "joint_positions.3": 0.0,
            "joint_positions.4": 0.0,
            "joint_positions.5": 0.0,
            "joint_positions.6": 0.0,
            "wrist_wrench.force.x": float(obs.wrist_wrench.wrench.force.x),
            "wrist_wrench.force.y": float(obs.wrist_wrench.wrench.force.y),
            "wrist_wrench.force.z": float(obs.wrist_wrench.wrench.force.z),
            "wrist_wrench.torque.x": float(obs.wrist_wrench.wrench.torque.x),
            "wrist_wrench.torque.y": float(obs.wrist_wrench.wrench.torque.y),
            "wrist_wrench.torque.z": float(obs.wrist_wrench.wrench.torque.z),
            "left_camera": self._image_to_bgr(obs.left_image),
            "center_camera": self._image_to_bgr(obs.center_image),
            "right_camera": self._image_to_bgr(obs.right_image),
        }

        joints = _fixed_len([float(v) for v in obs.joint_states.position], 7)
        for idx in range(7):
            values[f"joint_positions.{idx}"] = joints[idx]

        return values

    def _cartesian_action_from_motion(self) -> dict[str, float]:
        now = time.monotonic()
        zero_action = {
            "delta_position.x": 0.0,
            "delta_position.y": 0.0,
            "delta_position.z": 0.0,
            "delta_rotation.x": 0.0,
            "delta_rotation.y": 0.0,
            "delta_rotation.z": 0.0,
        }

        if self._latest_motion_cmd is None:
            return zero_action
        if (now - self._latest_motion_recv_time) > self.action_timeout_sec:
            return zero_action

        msg = self._latest_motion_cmd

        if (
            int(msg.trajectory_generation_mode.mode)
            != TrajectoryGenerationMode.MODE_POSITION
        ):
            mode_key = f"mode:{int(msg.trajectory_generation_mode.mode)}"
            if mode_key not in self._unsupported_action_modes_logged:
                self.get_logger().warn(
                    "Recorder expected Cartesian MODE_POSITION commands for delta-pose "
                    f"datasets, received mode={int(msg.trajectory_generation_mode.mode)}. "
                    "Recording zero action for this frame."
                )
                self._unsupported_action_modes_logged.add(mode_key)
            return zero_action

        frame_id = str(getattr(msg.header, "frame_id", "") or "")
        if frame_id != "gripper/tcp":
            mode_key = f"frame:{frame_id}"
            if mode_key not in self._unsupported_action_modes_logged:
                self.get_logger().warn(
                    "Recorder expected Cartesian pose commands in frame_id='gripper/tcp' "
                    f"for delta-pose datasets, received frame_id='{frame_id}'. "
                    "Recording zero action for this frame."
                )
                self._unsupported_action_modes_logged.add(mode_key)
            return zero_action

        rotvec = quaternion_xyzw_to_rotation_vector(
            np.array(
                [
                    float(msg.pose.orientation.x),
                    float(msg.pose.orientation.y),
                    float(msg.pose.orientation.z),
                    float(msg.pose.orientation.w),
                ],
                dtype=np.float64,
            )
        )
        return {
            "delta_position.x": float(msg.pose.position.x),
            "delta_position.y": float(msg.pose.position.y),
            "delta_position.z": float(msg.pose.position.z),
            "delta_rotation.x": float(rotvec[0]),
            "delta_rotation.y": float(rotvec[1]),
            "delta_rotation.z": float(rotvec[2]),
        }

    def _joint_action_from_joint_cmd(self) -> dict[str, float]:
        now = time.time()
        zero_action = {
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": 0.0,
            "elbow_joint": 0.0,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        }

        if self._latest_joint_cmd is None:
            return zero_action
        if (now - self._latest_joint_wall_time) > self.action_timeout_sec:
            return zero_action

        v = _fixed_len([float(x) for x in self._latest_joint_cmd.target_state.velocities], 6)
        return {
            "shoulder_pan_joint": v[0],
            "shoulder_lift_joint": v[1],
            "elbow_joint": v[2],
            "wrist_1_joint": v[3],
            "wrist_2_joint": v[4],
            "wrist_3_joint": v[5],
        }

    def _action_values(self) -> dict[str, float]:
        if self.action_mode == "joint":
            return self._joint_action_from_joint_cmd()
        return self._cartesian_action_from_motion()

    def _init_dataset(self, obs_values: dict[str, Any]) -> None:
        image_shape = tuple(int(x) for x in obs_values["left_camera"].shape)

        obs_state_features = {
            "tcp_pose.position.x": float,
            "tcp_pose.position.y": float,
            "tcp_pose.position.z": float,
            "tcp_pose.orientation.x": float,
            "tcp_pose.orientation.y": float,
            "tcp_pose.orientation.z": float,
            "tcp_pose.orientation.w": float,
            "tcp_velocity.linear.x": float,
            "tcp_velocity.linear.y": float,
            "tcp_velocity.linear.z": float,
            "tcp_velocity.angular.x": float,
            "tcp_velocity.angular.y": float,
            "tcp_velocity.angular.z": float,
            "tcp_error.x": float,
            "tcp_error.y": float,
            "tcp_error.z": float,
            "tcp_error.rx": float,
            "tcp_error.ry": float,
            "tcp_error.rz": float,
            "joint_positions.0": float,
            "joint_positions.1": float,
            "joint_positions.2": float,
            "joint_positions.3": float,
            "joint_positions.4": float,
            "joint_positions.5": float,
            "joint_positions.6": float,
            "wrist_wrench.force.x": float,
            "wrist_wrench.force.y": float,
            "wrist_wrench.force.z": float,
            "wrist_wrench.torque.x": float,
            "wrist_wrench.torque.y": float,
            "wrist_wrench.torque.z": float,
            "left_camera": image_shape,
            "center_camera": image_shape,
            "right_camera": image_shape,
        }

        action_features: dict[str, type]
        if self.action_mode == "joint":
            action_features = dict(JointMotionUpdateActionDict.__annotations__)
        else:
            action_features = dict(MotionUpdateActionDict.__annotations__)

        dataset_features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=self._teleop_action_processor,
                initial_features=create_initial_features(action=action_features),
                use_videos=self.video,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=self._robot_observation_processor,
                initial_features=create_initial_features(observation=obs_state_features),
                use_videos=self.video,
            ),
        )

        if self.resume:
            self._dataset = LeRobotDataset.resume(
                self.repo_id,
                root=self.root,
                batch_encoding_size=self.video_encoding_batch_size,
                vcodec=self.vcodec,
                image_writer_processes=self.image_writer_processes if self.video else 0,
                image_writer_threads=self.image_writer_threads_per_camera * 3 if self.video else 0,
            )
            sanity_check_dataset_robot_compatibility(
                self._dataset,
                _RobotTypeShim(self.robot_type),
                self.fps,
                dataset_features,
            )
            self.get_logger().info(
                f"Resuming dataset '{self.repo_id}' with {self._dataset.num_episodes} existing episodes."
            )
        else:
            self._dataset = LeRobotDataset.create(
                self.repo_id,
                self.fps,
                root=self.root,
                robot_type=self.robot_type,
                features=dataset_features,
                use_videos=self.video,
                image_writer_processes=self.image_writer_processes,
                image_writer_threads=self.image_writer_threads_per_camera * 3,
                batch_encoding_size=self.video_encoding_batch_size,
                vcodec=self.vcodec,
            )
            self.get_logger().info(f"Created dataset '{self.repo_id}'.")

        self._dataset_ready = True

    def _save_or_drop_episode(self, terminal_state: int) -> None:
        goal_id = self._current_goal_id
        if goal_id is None:
            return

        if self._dataset is None:
            self._set_goal_save_state(
                goal_id,
                finalized=True,
                saved=False,
                terminal_state=terminal_state,
                message="Recorder dataset is uninitialized; episode could not be saved.",
            )
            self._recording = False
            self._current_goal_id = None
            self._pending_terminal_state = None
            return

        success = terminal_state == GoalStatus.STATUS_SUCCEEDED
        success = True
        saved = False
        status_message = ""
        try:
            if success or self.save_failed_episodes:
                self._dataset.save_episode()
                self._episodes_saved += 1
                saved = True
                status_message = (
                    f"Episode saved (success={success}). total_saved={self._episodes_saved}"
                )
                self.get_logger().info(status_message)
            else:
                self._dataset.clear_episode_buffer()
                status_message = (
                    f"Episode dropped (terminal_state={terminal_state}) "
                    "because save_failed_episodes is disabled."
                )
                self.get_logger().info(status_message)
        except Exception as ex:
            saved = False
            status_message = (
                "Episode finalization failed during save/clear: "
                f"{type(ex).__name__}: {ex}"
            )
            self.get_logger().error(status_message)

        self._set_goal_save_state(
            goal_id,
            finalized=True,
            saved=saved,
            terminal_state=terminal_state,
            message=status_message,
        )

        self._recording = False
        self._current_goal_id = None
        self._pending_terminal_state = None

        if self.max_episodes > 0 and self._episodes_saved >= self.max_episodes:
            self.get_logger().info(
                f"Reached max episodes ({self.max_episodes}). Stopping recorder."
            )
            rclpy.shutdown()

    def _sample_frame(self) -> None:
        if self._latest_obs is None:
            return

        obs_values = self._obs_message_to_values(self._latest_obs)

        if not self._dataset_ready:
            self._init_dataset(obs_values)

        if not self._recording or self._dataset is None:
            return

        obs_values_processed = self._robot_observation_processor(obs_values)
        action_values_raw = self._action_values()
        action_values_processed = self._teleop_action_processor((action_values_raw, obs_values))

        observation_frame = build_dataset_frame(
            self._dataset.features,
            obs_values_processed,
            prefix=OBS_STR,
        )
        action_frame = build_dataset_frame(
            self._dataset.features,
            action_values_processed,
            prefix=ACTION,
        )

        frame = {**observation_frame, **action_frame, "task": self.single_task}
        self._dataset.add_frame(frame)

        if self._pending_terminal_state is not None:
            self._save_or_drop_episode(self._pending_terminal_state)

    def close(self) -> None:
        if self._dataset is not None:
            if self._recording and self._pending_terminal_state is None:
                self.get_logger().warn(
                    "Shutting down while episode is active. Dropping current episode buffer."
                )
                self._dataset.clear_episode_buffer()

            self._dataset.finalize()
            self.get_logger().info("Dataset finalized.")

            if self.push_to_hub:
                self._dataset.push_to_hub(tags=self.tags, private=self.private)
                self.get_logger().info("Dataset pushed to hub.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record policy rollouts into native LeRobot dataset format."
    )

    parser.add_argument("--dataset.repo_id", dest="repo_id", required=True)
    parser.add_argument("--dataset.single_task", dest="single_task", required=True)
    parser.add_argument("--dataset.root", dest="root", type=Path, default=None)
    parser.add_argument("--dataset.fps", dest="fps", type=int, default=30)
    parser.add_argument("--dataset.video", dest="video", action="store_true")
    parser.add_argument(
        "--no-dataset.video",
        dest="video",
        action="store_false",
        help="Disable video/image storage",
    )
    parser.set_defaults(video=True)
    parser.add_argument("--dataset.vcodec", dest="vcodec", default="libsvtav1")
    parser.add_argument(
        "--dataset.num_image_writer_processes",
        dest="num_image_writer_processes",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--dataset.num_image_writer_threads_per_camera",
        dest="num_image_writer_threads_per_camera",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--dataset.video_encoding_batch_size",
        dest="video_encoding_batch_size",
        type=int,
        default=1,
    )
    parser.add_argument("--dataset.resume", dest="resume", action="store_true")
    parser.add_argument(
        "--dataset.push_to_hub", dest="push_to_hub", action="store_true", default=False
    )
    parser.add_argument(
        "--dataset.private", dest="private", action="store_true", default=True
    )
    parser.add_argument(
        "--dataset.tags",
        dest="tags",
        default="",
        help="Comma-separated tags for push_to_hub",
    )

    parser.add_argument(
        "--save_failed_episodes",
        action="store_true",
        help="Save failed/canceled episodes instead of dropping them.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=0,
        help="Stop after this many saved episodes (0 = unlimited).",
    )

    parser.add_argument(
        "--action_mode",
        choices=["cartesian", "joint"],
        default="cartesian",
        help="Action schema for dataset. Must match teleop dataset mode for seamless merge.",
    )
    parser.add_argument(
        "--action_timeout_sec",
        type=float,
        default=0.4,
        help="How long a command stays valid before action is zeroed.",
    )
    parser.add_argument(
        "--camera_scale",
        type=float,
        default=0.25,
        help="Image scale factor to match teleop datasets (default 0.25).",
    )

    parser.add_argument("--robot_type", default="ur5e_aic")

    parser.add_argument("--observations_topic", default="/observations")
    parser.add_argument("--pose_commands_topic", default="/aic_controller/pose_commands")
    parser.add_argument("--joint_commands_topic", default="/aic_controller/joint_commands")
    parser.add_argument("--status_topic", default="/insert_cable/_action/status")
    parser.add_argument(
        "--status_service_name",
        default="/aic_policy_recorder/get_episode_save_status",
        help="Service used by aic_engine to query per-goal recorder save status.",
    )

    args = parser.parse_args()
    if args.fps <= 0:
        raise ValueError("--dataset.fps must be > 0")
    if args.action_timeout_sec <= 0:
        raise ValueError("--action_timeout_sec must be > 0")
    if args.camera_scale <= 0:
        raise ValueError("--camera_scale must be > 0")

    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    tags = [x.strip() for x in args.tags.split(",") if x.strip()] if args.tags else None

    with rclpy.init():
        node = PolicyRecorder(
            repo_id=args.repo_id,
            single_task=args.single_task,
            root=args.root,
            fps=args.fps,
            video=args.video,
            vcodec=args.vcodec,
            image_writer_processes=args.num_image_writer_processes,
            image_writer_threads_per_camera=args.num_image_writer_threads_per_camera,
            video_encoding_batch_size=args.video_encoding_batch_size,
            resume=args.resume,
            push_to_hub=args.push_to_hub,
            private=args.private,
            tags=tags,
            save_failed_episodes=args.save_failed_episodes,
            max_episodes=args.max_episodes,
            action_mode=args.action_mode,
            action_timeout_sec=args.action_timeout_sec,
            camera_scale=args.camera_scale,
            observations_topic=args.observations_topic,
            pose_commands_topic=args.pose_commands_topic,
            joint_commands_topic=args.joint_commands_topic,
            status_topic=args.status_topic,
            status_service_name=args.status_service_name,
            robot_type=args.robot_type,
        )
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.close()
            node.destroy_node()


if __name__ == "__main__":
    main()
