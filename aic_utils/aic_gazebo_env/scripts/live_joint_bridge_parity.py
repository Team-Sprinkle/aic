#!/usr/bin/env python3
"""Live validation and parity harness for the Gazebo joint bridge."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import rclpy
from aic_control_interfaces.msg import ControllerState, JointMotionUpdate, TargetMode
from aic_control_interfaces.srv import ChangeTargetMode
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from aic_gazebo_env.gazebo_client import GazeboCliClient, GazeboCliClientConfig
from aic_gazebo_env.protocol import GetObservationRequest, ResetRequest, StepRequest


JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
POLICY_DELTAS: list[list[float]] = [
    [0.02, -0.01, 0.015, 0.0, 0.0, 0.0],
    [0.02, -0.01, 0.015, 0.0, 0.0, 0.0],
    [0.02, -0.01, 0.015, 0.0, 0.0, 0.0],
    [0.02, -0.01, 0.015, 0.0, 0.0, 0.0],
    [0.02, -0.01, 0.015, 0.0, 0.0, 0.0],
]
JOINT_TOLERANCE = 1e-4
POSITION_TOLERANCE = 1e-4
ORIENTATION_TOLERANCE = 1e-3
TOOLKIT_SETTLE_TOLERANCE = 5e-2
TOOLKIT_SETTLE_TIMEOUT_S = 2.0


@dataclass
class StepTrace:
    step_index: int
    joint_positions: list[float]
    tcp_position: list[float]
    relative_position: list[float]
    distance: float
    orientation_error: float
    reward: float
    terminated: bool
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "joint_positions": self.joint_positions,
            "tcp_position": self.tcp_position,
            "relative_position": self.relative_position,
            "distance": self.distance,
            "orientation_error": self.orientation_error,
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }


class ToolkitDriver(Node):
    """Drive the official controller path inside a running bringup."""

    def __init__(self, client: GazeboCliClient) -> None:
        super().__init__("live_joint_bridge_parity")
        self._client = client
        self._joint_state: JointState | None = None
        self._controller_state: ControllerState | None = None
        self._change_mode = self.create_client(
            ChangeTargetMode,
            "/aic_controller/change_target_mode",
        )
        self._publisher = self.create_publisher(
            JointMotionUpdate,
            "/aic_controller/joint_commands",
            10,
        )
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        self.create_subscription(
            ControllerState,
            "/aic_controller/controller_state",
            self._on_controller_state,
            10,
        )

    def _on_joint_state(self, msg: JointState) -> None:
        self._joint_state = msg

    def _on_controller_state(self, msg: ControllerState) -> None:
        self._controller_state = msg

    def wait_for_ros(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._change_mode.wait_for_service(timeout_sec=0.1):
                rclpy.spin_once(self, timeout_sec=0.1)
                if self._joint_state is not None:
                    return
            else:
                rclpy.spin_once(self, timeout_sec=0.1)
        raise RuntimeError("Timed out waiting for ROS controller interfaces.")

    def reset(self) -> dict[str, Any]:
        self._client.reset(ResetRequest(seed=0, options={"mode": "live-parity"}))
        self._set_joint_mode()
        return self._client.get_observation(GetObservationRequest()).observation

    def step(self, joint_delta: list[float], step_index: int) -> StepTrace:
        current_observation = self._client.get_observation(GetObservationRequest()).observation
        current_joint_positions = list(current_observation["joint_positions"])
        target_positions = [
            current_position + delta
            for current_position, delta in zip(current_joint_positions, joint_delta)
        ]
        self._publish_joint_target(target_positions)
        self._wait_for_joint_target(target_positions)
        observation = self._client.get_observation(GetObservationRequest()).observation
        reward, terminated, truncated, _ = self._client._compute_step_outcome(
            observation
        )
        return trace_from_observation(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            step_index=step_index,
        )

    def _set_joint_mode(self) -> None:
        request = ChangeTargetMode.Request()
        request.target_mode.mode = TargetMode.MODE_JOINT
        future = self._change_mode.call_async(request)
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
        response = future.result()
        if response is None or not response.success:
            raise RuntimeError("Failed to switch aic_controller into joint mode.")

    def _publish_joint_target(self, target_positions: list[float]) -> None:
        msg = JointMotionUpdate()
        msg.target_state = JointTrajectoryPoint()
        msg.target_state.positions = target_positions
        msg.target_stiffness = [85.0] * len(JOINT_NAMES)
        msg.target_damping = [75.0] * len(JOINT_NAMES)
        msg.trajectory_generation_mode.mode = 2
        self._publisher.publish(msg)

    def _wait_for_joint_target(self, target_positions: list[float]) -> None:
        deadline = time.monotonic() + TOOLKIT_SETTLE_TIMEOUT_S
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._joint_state is None:
                continue
            positions_by_name = {
                name: position
                for name, position in zip(self._joint_state.name, self._joint_state.position)
            }
            actual_positions = [positions_by_name.get(name) for name in JOINT_NAMES]
            if not all(isinstance(value, float) for value in actual_positions):
                continue
            max_error = max(
                abs(target - actual)
                for target, actual in zip(target_positions, actual_positions)
            )
            if max_error <= TOOLKIT_SETTLE_TOLERANCE:
                return
        raise RuntimeError(
            "Toolkit controller did not settle to the requested target within "
            f"{TOOLKIT_SETTLE_TIMEOUT_S:.2f}s."
        )


def run_command(command: list[str], *, timeout: float = 5.0) -> str:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {command!r}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed.stdout.strip()


def discover_joint_target_service() -> tuple[str, str]:
    output = run_command(["gz", "service", "-l"], timeout=10.0)
    matches = [
        line.strip()
        for line in output.splitlines()
        if line.strip().endswith("/joint_target")
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one joint_target service, found {len(matches)}: {matches}"
        )
    service_path = matches[0]
    world_name = service_path.split("/")[2]
    return world_name, service_path


def build_client(world_name: str) -> GazeboCliClient:
    return GazeboCliClient(
        GazeboCliClientConfig(
            executable="gz",
            world_path=str(Path("aic_description/world/aic.sdf").resolve()),
            timeout=10.0,
            world_name=world_name,
            source_entity_name="gripper/tcp",
            target_entity_name="task_board",
            joint_command_model_name="ur",
            joint_names=JOINT_NAMES,
        )
    )


def trace_from_observation(
    *,
    observation: dict[str, Any],
    reward: float,
    terminated: bool,
    truncated: bool,
    step_index: int,
) -> StepTrace:
    tracked_pair = observation["task_geometry"]["tracked_entity_pair"]
    tcp_position = observation["entities_by_name"]["gripper/tcp"]["position"]
    return StepTrace(
        step_index=step_index,
        joint_positions=list(observation["joint_positions"]),
        tcp_position=list(tcp_position),
        relative_position=list(tracked_pair["relative_position"]),
        distance=float(tracked_pair["distance"]),
        orientation_error=float(tracked_pair["orientation_error"]),
        reward=float(reward),
        terminated=bool(terminated),
        truncated=bool(truncated),
    )


def run_env_rollout(client: GazeboCliClient) -> list[StepTrace]:
    client.reset(ResetRequest(seed=0, options={"mode": "live-parity"}))
    traces: list[StepTrace] = []
    for step_index, joint_delta in enumerate(POLICY_DELTAS):
        response = client.step(
            StepRequest(
                action={
                    "joint_position_delta": list(joint_delta),
                    "multi_step": 1,
                }
            )
        )
        traces.append(
            trace_from_observation(
                observation=response.observation,
                reward=response.reward,
                terminated=response.terminated,
                truncated=response.truncated,
                step_index=step_index,
            )
        )
        if response.terminated or response.truncated:
            break
    return traces


def run_toolkit_rollout(driver: ToolkitDriver) -> list[StepTrace]:
    driver.reset()
    traces: list[StepTrace] = []
    for step_index, joint_delta in enumerate(POLICY_DELTAS):
        trace = driver.step(joint_delta, step_index)
        traces.append(trace)
        if trace.terminated or trace.truncated:
            break
    return traces


def max_abs_diff(left: list[float], right: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(left, right))


def compare_traces(toolkit: list[StepTrace], env: list[StepTrace]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    compared_steps = min(len(toolkit), len(env))
    for index in range(compared_steps):
        toolkit_step = toolkit[index]
        env_step = env[index]
        joint_diff = max_abs_diff(toolkit_step.joint_positions, env_step.joint_positions)
        tcp_diff = max_abs_diff(toolkit_step.tcp_position, env_step.tcp_position)
        relative_diff = max_abs_diff(
            toolkit_step.relative_position,
            env_step.relative_position,
        )
        distance_diff = abs(toolkit_step.distance - env_step.distance)
        orientation_diff = abs(
            toolkit_step.orientation_error - env_step.orientation_error
        )
        reward_diff = abs(toolkit_step.reward - env_step.reward)
        flags_match = (
            toolkit_step.terminated == env_step.terminated
            and toolkit_step.truncated == env_step.truncated
        )
        within_tolerance = (
            joint_diff <= JOINT_TOLERANCE
            and tcp_diff <= POSITION_TOLERANCE
            and relative_diff <= POSITION_TOLERANCE
            and distance_diff <= POSITION_TOLERANCE
            and orientation_diff <= ORIENTATION_TOLERANCE
            and reward_diff <= POSITION_TOLERANCE
            and flags_match
        )
        if not within_tolerance:
            mismatches.append(
                {
                    "step_index": index,
                    "joint_diff": joint_diff,
                    "tcp_diff": tcp_diff,
                    "relative_diff": relative_diff,
                    "distance_diff": distance_diff,
                    "orientation_diff": orientation_diff,
                    "reward_diff": reward_diff,
                    "toolkit": toolkit_step.to_dict(),
                    "env": env_step.to_dict(),
                }
            )
            break
    return {
        "toolkit_steps": len(toolkit),
        "env_steps": len(env),
        "compared_steps": compared_steps,
        "match": not mismatches and len(toolkit) == len(env),
        "first_mismatch": mismatches[0] if mismatches else None,
    }


def benchmark_rollout(label: str, rollout_fn: Any) -> dict[str, Any]:
    start = time.perf_counter()
    traces = rollout_fn()
    elapsed = time.perf_counter() - start
    steps = len(traces)
    return {
        "label": label,
        "steps": steps,
        "elapsed_s": elapsed,
        "steps_per_s": steps / elapsed if elapsed > 0 else math.inf,
    }


def phase_header(index: int, title: str) -> None:
    print(f"\nPhase {index}: {title}")


def print_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def main() -> int:
    phase_header(1, "live service path fix/validation")
    print("COMMAND: gz service -l | grep joint_target")
    world_name, service_path = discover_joint_target_service()
    print("OUTPUT:")
    print(service_path)
    print("PASS")
    print_json(
        {
            "world_name": world_name,
            "joint_target_service": service_path,
            "next_action": "send one real joint target and verify joints plus TCP pose move",
        }
    )

    client = build_client(world_name)

    phase_header(2, "real joint command changes state")
    print(
        "COMMAND: PYTHONPATH=aic_utils/aic_gazebo_env "
        "python3 aic_utils/aic_gazebo_env/scripts/live_joint_bridge_parity.py"
    )
    reset_response = client.reset(ResetRequest(seed=0, options={"mode": "live-phase-2"}))
    pre = trace_from_observation(
        observation=reset_response.observation,
        reward=0.0,
        terminated=False,
        truncated=False,
        step_index=0,
    )
    step_response = client.step(
        StepRequest(
            action={
                "joint_position_delta": list(POLICY_DELTAS[0]),
                "multi_step": 1,
            }
        )
    )
    post = trace_from_observation(
        observation=step_response.observation,
        reward=step_response.reward,
        terminated=step_response.terminated,
        truncated=step_response.truncated,
        step_index=1,
    )
    phase_2_pass = (
        max_abs_diff(pre.joint_positions, post.joint_positions) > 0.0
        and max_abs_diff(pre.tcp_position, post.tcp_position) > 0.0
        and step_response.info.get("pose_service") is None
    )
    print("OUTPUT:")
    print_json(
        {
            "pre": pre.to_dict(),
            "post": post.to_dict(),
            "joint_target_service_reply": step_response.info.get("joint_target_service"),
            "pose_service_reply": step_response.info.get("pose_service"),
        }
    )
    print("PASS" if phase_2_pass else "FAIL")
    if not phase_2_pass:
        return 1

    phase_header(3, "parity harness and comparison")
    rclpy.init(args=None)
    toolkit_driver = ToolkitDriver(client)
    try:
        toolkit_driver.wait_for_ros(timeout_s=10.0)
        toolkit_traces = run_toolkit_rollout(toolkit_driver)
        env_traces = run_env_rollout(client)
    finally:
        toolkit_driver.destroy_node()
        rclpy.shutdown()
    comparison = compare_traces(toolkit_traces, env_traces)
    print("OUTPUT:")
    print_json(
        {
            "policy_joint_deltas": POLICY_DELTAS,
            "toolkit_trace": [trace.to_dict() for trace in toolkit_traces],
            "env_trace": [trace.to_dict() for trace in env_traces],
            "comparison": comparison,
        }
    )
    print("PASS" if comparison["match"] else "FAIL")

    phase_header(4, "mismatches and fixes, if any")
    if comparison["match"]:
        print("PASS")
        print_json({"message": "No parity mismatch detected for the chosen stable slice."})
    else:
        first_mismatch = comparison["first_mismatch"]
        classification = "controller/plugin mismatch"
        if first_mismatch is not None and first_mismatch["joint_diff"] > JOINT_TOLERANCE:
            classification = "action interpretation mismatch"
        print("FAIL")
        print_json(
            {
                "first_mismatch": first_mismatch,
                "classification": classification,
                "next_action": "fix the smallest cause, then rerun the same harness",
            }
        )

    phase_header(5, "benchmark and speedup")
    rclpy.init(args=None)
    toolkit_driver = ToolkitDriver(client)
    try:
        toolkit_driver.wait_for_ros(timeout_s=10.0)
        toolkit_benchmark = benchmark_rollout(
            "toolkit_ros_controller_path",
            lambda: run_toolkit_rollout(toolkit_driver),
        )
    finally:
        toolkit_driver.destroy_node()
        rclpy.shutdown()
    env_benchmark = benchmark_rollout("gazebo_joint_bridge_path", lambda: run_env_rollout(client))
    speed_ratio = (
        env_benchmark["steps_per_s"] / toolkit_benchmark["steps_per_s"]
        if toolkit_benchmark["steps_per_s"] not in (0.0, math.inf)
        else math.inf
    )
    print("OUTPUT:")
    print_json(
        {
            "toolkit": toolkit_benchmark,
            "env": env_benchmark,
            "relative_speedup_env_vs_toolkit": speed_ratio,
        }
    )
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
