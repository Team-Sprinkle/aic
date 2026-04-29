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

import numpy as np
from abc import ABC, abstractmethod
from aic_control_interfaces.msg import (
    JointMotionUpdate,
    MotionUpdate,
    TrajectoryGenerationMode,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, Wrench
from rclpy.duration import Duration
from std_msgs.msg import Header
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from typing import Callable, Protocol

GetObservationCallback = Callable[[], Observation]


class MoveRobotCallback(Protocol):
    """Move the robot using either Cartesian or joint-space commands.

    This function is called by a policy to request robot motion. Either
    Cartesian or joint-space commands can be sent, but not both at the
    same time. One of the following must be set:
     - motion_update: cartesian motion commands
     - joint_motion_update: joint-space motion commands

    The MotionUpdate message contains a request to the Cartesian-space
    admittance controller. The details of this message are described
    in its message definition:
      https://github.com/intrinsic-dev/aic/blob/main/aic_interfaces/aic_control_interfaces/msg/MotionUpdate.msg

    The JointMotionUpdate message contains commands to the joint-space
    controller. The details of this message are described in its message definition:
      https://github.com/intrinsic-dev/aic/blob/main/aic_interfaces/aic_control_interfaces/msg/JointMotionUpdate.msg

    As a convenience, a reasonable set of parameters is populated in a MotionUpdate
    message by the create_motion_update(pose) function in the Policy class.
    """

    def __call__(
        self,
        motion_update: MotionUpdate = None,
        joint_motion_update: JointMotionUpdate = None,
    ) -> None: ...


SendFeedbackCallback = Callable[[str], None]

DEFAULT_CARTESIAN_STIFFNESS = [90.0, 90.0, 90.0, 50.0, 50.0, 50.0]
DEFAULT_CARTESIAN_DAMPING = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]


def _as_np_vector3(vector: Vector3 | Point) -> np.ndarray:
    return np.array([float(vector.x), float(vector.y), float(vector.z)], dtype=np.float64)


def _as_np_quat_xyzw(quat: Quaternion) -> np.ndarray:
    return np.array(
        [float(quat.x), float(quat.y), float(quat.z), float(quat.w)],
        dtype=np.float64,
    )


def normalize_quaternion_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quat_xyzw))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    quat = quat_xyzw / norm
    if quat[3] < 0.0:
        quat = -quat
    return quat


def quaternion_xyzw_to_rotation_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quaternion_xyzw(quat_xyzw)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def quaternion_multiply_xyzw(q1_xyzw: np.ndarray, q2_xyzw: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1_xyzw
    x2, y2, z2, w2 = q2_xyzw
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def quaternion_inverse_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = normalize_quaternion_xyzw(quat_xyzw)
    return np.array([-quat[0], -quat[1], -quat[2], quat[3]], dtype=np.float64)


def rotation_vector_to_quaternion_xyzw(rotvec_xyz: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec_xyz, dtype=np.float64)
    angle = float(np.linalg.norm(rotvec))
    if angle <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = rotvec / angle
    half_angle = 0.5 * angle
    sin_half = np.sin(half_angle)
    quat = np.array(
        [
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half,
            np.cos(half_angle),
        ],
        dtype=np.float64,
    )
    return normalize_quaternion_xyzw(quat)


def quaternion_xyzw_to_rotation_vector(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = normalize_quaternion_xyzw(quat_xyzw)
    xyz = quat[:3]
    sin_half = float(np.linalg.norm(xyz))
    if sin_half <= 1e-12:
        return 2.0 * xyz
    half_angle = np.arctan2(sin_half, float(np.clip(quat[3], -1.0, 1.0)))
    return xyz / sin_half * (2.0 * half_angle)


def build_pose_from_vectors(
    position_xyz: np.ndarray, orientation_xyzw: np.ndarray
) -> Pose:
    quat = normalize_quaternion_xyzw(np.asarray(orientation_xyzw, dtype=np.float64))
    pos = np.asarray(position_xyz, dtype=np.float64)
    return Pose(
        position=Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
        orientation=Quaternion(
            x=float(quat[0]),
            y=float(quat[1]),
            z=float(quat[2]),
            w=float(quat[3]),
        ),
    )


def pose_to_position_motion_update(
    pose: Pose,
    *,
    stamp,
    frame_id: str,
    stiffness: list[float],
    damping: list[float],
) -> MotionUpdate:
    return MotionUpdate(
        header=Header(frame_id=frame_id, stamp=stamp),
        pose=pose,
        target_stiffness=np.diag(stiffness).flatten(),
        target_damping=np.diag(damping).flatten(),
        feedforward_wrench_at_tip=Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        ),
        wrench_feedback_gains_at_tip=[0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
        trajectory_generation_mode=TrajectoryGenerationMode(
            mode=TrajectoryGenerationMode.MODE_POSITION,
        ),
    )


def compute_delta_pose(current_pose: Pose, target_pose: Pose) -> Pose:
    current_pos = _as_np_vector3(current_pose.position)
    target_pos = _as_np_vector3(target_pose.position)
    current_quat = _as_np_quat_xyzw(current_pose.orientation)
    target_quat = _as_np_quat_xyzw(target_pose.orientation)

    rot_world_to_current = quaternion_xyzw_to_rotation_matrix(
        quaternion_inverse_xyzw(current_quat)
    )
    delta_pos = rot_world_to_current @ (target_pos - current_pos)
    delta_quat = quaternion_multiply_xyzw(
        quaternion_inverse_xyzw(current_quat),
        target_quat,
    )
    return build_pose_from_vectors(delta_pos, delta_quat)


def clamp_delta_pose_components(
    position_xyz: np.ndarray,
    rotation_xyz: np.ndarray,
    *,
    max_translation: float | None = None,
    max_rotation: float | None = None,
    deadband_translation: float = 0.0,
    deadband_rotation: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(position_xyz, dtype=np.float64).copy()
    rotation = np.asarray(rotation_xyz, dtype=np.float64).copy()

    if deadband_translation > 0.0:
        position[np.abs(position) < deadband_translation] = 0.0
    if deadband_rotation > 0.0:
        rotation[np.abs(rotation) < deadband_rotation] = 0.0

    if max_translation is not None:
        position = np.clip(position, -max_translation, max_translation)
    if max_rotation is not None:
        rotation = np.clip(rotation, -max_rotation, max_rotation)

    return position, rotation


class Policy(ABC):
    def __init__(self, parent_node):
        self._parent_node = parent_node
        self.get_logger().info("Policy.__init__()")

    def get_logger(self):
        return self._parent_node.get_logger()

    def get_clock(self):
        return self._parent_node.get_clock()

    def time_now(self):
        """Return the current time from the node's clock (sim-time aware)."""
        return self.get_clock().now()

    def sleep_for(self, duration_sec: float) -> None:
        """Sleep for the given duration using the node's clock (sim-time aware)."""
        self.get_clock().sleep_for(Duration(seconds=duration_sec))

    def set_pose_target(
        self,
        move_robot: MoveRobotCallback,
        pose: Pose,
        frame_id: str = "base_link",
        stiffness: list = DEFAULT_CARTESIAN_STIFFNESS,
        damping: list = DEFAULT_CARTESIAN_DAMPING,
    ) -> None:
        """Invoke the move_robot callback to request the supplied Pose.

        This is a convenience function which populates a MotionUpdate message
        with a reasonable set of default parameters, and invokes the move_robot
        callback to request motion to the supplied pose.

        The robot can be controlled in several different ways. This function
        is intended to be the simplest way to move the arm around, by sending
        a desired pose (position and orientation) for the gripper's
        "tool control point" (TCP), which is the "pinch point" between the very
        end of the gripper fingers. The rest of the control stack will take care
        of moving all the arm's joints to so that the gripper TCP ends up in
        the desired position and orientation.

        The constants defined in this function are intended to provide
        reasonable default behavior if the arm is unable to achieve the
        requested pose. Different values for stiffness, damping, wrenches, and
        so on can be used for different types of arm behavior. These values
        are only intended to provide a starting point, and can be adjusted as
        desired.
        """
        motion_update = pose_to_position_motion_update(
            pose,
            stamp=self._parent_node.get_clock().now().to_msg(),
            frame_id=frame_id,
            stiffness=stiffness,
            damping=damping,
        )
        try:
            move_robot(motion_update=motion_update)
        except Exception as ex:
            self.get_logger().info(f"move_robot exception: {ex}")

    def set_delta_pose_target(
        self,
        move_robot: MoveRobotCallback,
        delta_pose: Pose,
        frame_id: str = "gripper/tcp",
        stiffness: list = DEFAULT_CARTESIAN_STIFFNESS,
        damping: list = DEFAULT_CARTESIAN_DAMPING,
    ) -> None:
        self.set_pose_target(
            move_robot=move_robot,
            pose=delta_pose,
            frame_id=frame_id,
            stiffness=stiffness,
            damping=damping,
        )

    def set_delta_pose_target_from_components(
        self,
        move_robot: MoveRobotCallback,
        delta_position_xyz: np.ndarray,
        delta_rotation_xyz: np.ndarray,
        *,
        frame_id: str = "gripper/tcp",
        stiffness: list = DEFAULT_CARTESIAN_STIFFNESS,
        damping: list = DEFAULT_CARTESIAN_DAMPING,
        max_translation: float | None = None,
        max_rotation: float | None = None,
        deadband_translation: float = 0.0,
        deadband_rotation: float = 0.0,
    ) -> None:
        position, rotation = clamp_delta_pose_components(
            delta_position_xyz,
            delta_rotation_xyz,
            max_translation=max_translation,
            max_rotation=max_rotation,
            deadband_translation=deadband_translation,
            deadband_rotation=deadband_rotation,
        )
        delta_pose = build_pose_from_vectors(
            position,
            rotation_vector_to_quaternion_xyzw(rotation),
        )
        self.set_delta_pose_target(
            move_robot=move_robot,
            delta_pose=delta_pose,
            frame_id=frame_id,
            stiffness=stiffness,
            damping=damping,
        )

    @abstractmethod
    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        """Called when the insert_cable task is requested by aic_engine"""
        pass
