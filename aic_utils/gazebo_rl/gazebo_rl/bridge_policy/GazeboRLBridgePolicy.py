from __future__ import annotations

import os
from typing import Any

from aic_model.policy import GetObservationCallback, MoveRobotCallback, Policy, SendFeedbackCallback
from aic_task_interfaces.msg import Task
from std_msgs.msg import String

from gazebo_rl.action import DEFAULT_MAX_ROTATION_RAD, DEFAULT_MAX_TRANSLATION_M, delta_tcp_action_from_array
from gazebo_rl.ipc import IPCError, JsonLineConnection, connect_with_retry
from gazebo_rl.observation import observation_to_dict


class GazeboRLBridgePolicy(Policy):
    """AIC policy that delegates each motion tick to an external RL trainer."""

    def __init__(self, parent_node):
        self._connection: JsonLineConnection | None = None
        self._task: Task | None = None
        self._step_count = 0
        self._latest_insertion_event_namespace = ""
        super().__init__(parent_node)
        self._insertion_event_sub = self._parent_node.create_subscription(
            String,
            "/scoring/insertion_event",
            self._insertion_event_callback,
            10,
        )

    def _env_float(self, name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, str(default)))
        except ValueError:
            self.get_logger().warn(f"Invalid {name}; using default {default}")
            return default

    def _env_int(self, name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, str(default)))
        except ValueError:
            self.get_logger().warn(f"Invalid {name}; using default {default}")
            return default

    def _env_bool(self, name: str, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    def _insertion_event_callback(self, msg: String) -> None:
        self._latest_insertion_event_namespace = msg.data.strip().strip("/")
        self.get_logger().info(
            f"Received insertion event for namespace: '{self._latest_insertion_event_namespace}'"
        )

    def _task_completed_in_simulation(self, task: Task) -> bool:
        namespace = self._latest_insertion_event_namespace
        if not namespace:
            return False
        tokens = [token for token in namespace.split("/") if token]
        if len(tokens) < 2:
            return False
        return tokens[-2] == task.target_module_name and tokens[-1] == task.port_name

    def _connect(self, task: Task) -> JsonLineConnection:
        host = os.environ.get("AIC_GAZEBO_RL_HOST", "127.0.0.1")
        port = self._env_int("AIC_GAZEBO_RL_PORT", 8765)
        timeout = self._env_float("AIC_GAZEBO_RL_CONNECT_TIMEOUT_SEC", 60.0)
        conn = connect_with_retry(host, port, timeout_sec=timeout)
        conn.send(
            "hello",
            {
                "policy": "gazebo_rl.bridge_policy.GazeboRLBridgePolicy",
                "action_space": "relative_tcp_delta",
                "action_shape": [6],
                "max_translation_m": DEFAULT_MAX_TRANSLATION_M,
                "max_rotation_rad": DEFAULT_MAX_ROTATION_RAD,
                "task": {
                    "id": task.id,
                    "cable_name": task.cable_name,
                    "plug_name": task.plug_name,
                    "target_module_name": task.target_module_name,
                    "port_name": task.port_name,
                },
            },
        )
        return conn

    def _send_error(self, message: str) -> None:
        if self._connection is None:
            return
        try:
            self._connection.send("error", {"message": message, "step_count": self._step_count})
        except Exception:
            pass

    def _send_delta_action(self, move_robot: MoveRobotCallback, raw_action: Any) -> None:
        delta = delta_tcp_action_from_array(raw_action)
        self.set_delta_pose_target_from_components(
            move_robot=move_robot,
            delta_position_xyz=delta.delta_position_xyz,
            delta_rotation_xyz=delta.delta_rotation_xyz,
            frame_id="gripper/tcp",
            max_translation=DEFAULT_MAX_TRANSLATION_M,
            max_rotation=DEFAULT_MAX_ROTATION_RAD,
        )

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self._task = task
        self._step_count = 0
        self._latest_insertion_event_namespace = ""
        command_dt_sec = self._env_float("AIC_GAZEBO_RL_COMMAND_DT_SEC", 0.05)
        max_steps = self._env_int("AIC_GAZEBO_RL_MAX_STEPS", 50)
        action_timeout_sec = self._env_float("AIC_GAZEBO_RL_ACTION_TIMEOUT_SEC", 30.0)
        ground_truth = self._env_bool("AIC_GAZEBO_RL_GROUND_TRUTH", False)

        self.get_logger().info(
            f"GazeboRLBridgePolicy starting task {task.id}; max_steps={max_steps}, dt={command_dt_sec}"
        )

        try:
            self._connection = self._connect(task)
        except Exception as ex:
            send_feedback(f"Gazebo RL IPC connection failed: {ex}")
            self.get_logger().error(f"Gazebo RL IPC connection failed: {ex}")
            return False

        succeeded = False
        try:
            while self._step_count < max_steps:
                if self._task_completed_in_simulation(task):
                    succeeded = True
                    self._connection.send("done", {"reason": "insertion_event", "step_count": self._step_count})
                    break

                try:
                    obs_msg = get_observation()
                    obs = observation_to_dict(
                        obs_msg,
                        task=task,
                        step_count=self._step_count,
                        tf_buffer=getattr(self._parent_node, "_tf_buffer", None),
                        ground_truth=ground_truth,
                        logger=lambda msg: self.get_logger().warn(msg),
                    )
                except Exception as ex:
                    send_feedback(f"Observation conversion failed at step {self._step_count}: {ex}")
                    self.get_logger().warn(f"Observation conversion failed, skipping tick: {ex}")
                    self.sleep_for(command_dt_sec)
                    continue

                self._connection.send(
                    "observation",
                    {
                        "observation": obs,
                        "step_count": self._step_count,
                        "terminated": False,
                    },
                )

                try:
                    message = self._connection.recv(timeout_sec=action_timeout_sec)
                except TimeoutError as ex:
                    send_feedback(str(ex))
                    self._send_error(str(ex))
                    break

                if message.type == "done":
                    self.get_logger().info(f"Trainer ended rollout: {message.payload}")
                    break
                if message.type != "action":
                    raise IPCError(f"Expected action message, received {message.type}")

                try:
                    self._send_delta_action(move_robot, message.payload.get("action"))
                except Exception as ex:
                    send_feedback(f"Invalid Gazebo RL action at step {self._step_count}: {ex}")
                    self._send_error(str(ex))
                    break

                self._step_count += 1
                self.sleep_for(command_dt_sec)

            else:
                self._connection.send("done", {"reason": "max_steps", "step_count": self._step_count})
        except (IPCError, OSError) as ex:
            send_feedback(f"Gazebo RL IPC disconnected: {ex}")
            self.get_logger().warn(f"Gazebo RL IPC disconnected: {ex}")
        except Exception as ex:
            send_feedback(f"Gazebo RL bridge error: {ex}")
            self.get_logger().error(f"Gazebo RL bridge error: {ex}")
            self._send_error(str(ex))
        finally:
            if self._connection is not None:
                try:
                    self._connection.close()
                except Exception:
                    pass
                self._connection = None

        self.get_logger().info(
            f"GazeboRLBridgePolicy exiting task {task.id}; steps={self._step_count}, succeeded={succeeded}"
        )
        return True
