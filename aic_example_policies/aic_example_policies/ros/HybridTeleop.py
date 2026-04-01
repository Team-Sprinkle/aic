import importlib
import inspect
import numpy as np
import queue
import threading

from aic_control_interfaces.msg import (
    JointMotionUpdate,
    MotionUpdate,
    TrajectoryGenerationMode,
)

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Pose, Twist, Vector3, Wrench
from std_msgs.msg import Header


class HybridTeleop(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)

        parent_node.declare_parameter(
            "hybrid.wrapped_policy", "aic_example_policies.ros.CheatCode"
        )
        parent_node.declare_parameter("hybrid.teleop_type", "keyboard_ee")

        wrapped_policy_module_name = (
            parent_node.get_parameter("hybrid.wrapped_policy")
            .get_parameter_value()
            .string_value
        )

        self._teleop_mode = (
            parent_node.get_parameter("hybrid.teleop_type")
            .get_parameter_value()
            .string_value
        )

        self.get_logger().info(
            f"HybridTeleop: wrapped_policy={wrapped_policy_module_name}, teleop_type={self._teleop_mode}"
        )

        try:
            wrapped_policy_module = importlib.import_module(wrapped_policy_module_name)
        except Exception as e:
            self.get_logger().error(
                f"Unable to load wrapped policy {wrapped_policy_module_name}: {e}"
            )
            raise e

        self.get_logger().info(
            f"Loaded wrapped policy module: {wrapped_policy_module_name}"
        )

        policy_module_classes = inspect.getmembers(
            wrapped_policy_module, inspect.isclass
        )
        self._wrapped_policy_class = None
        expected_policy_class_name = wrapped_policy_module_name.split(".")[-1]

        for policy_class_name, policy_class in policy_module_classes:
            if policy_class_name == expected_policy_class_name:
                self.get_logger().info(
                    f"Found wrapped policy class: {policy_class_name} in module {wrapped_policy_module_name}"
                )
                self._wrapped_policy_class = policy_class
                break
        if self._wrapped_policy_class is None:
            raise ValueError(
                f"Could not find policy class '{expected_policy_class_name}' in module '{wrapped_policy_module_name}'"
            )

        self._wrapped_policy = None
        self._teleop_instance = None
        # Threading and State
        self._active_source = "policy"  # or "teleop"
        self._source_lock = threading.Lock()
        self._policy_thread = None
        self._policy_thread_result = None
        # Signaling Events
        self._teleop_loop_should_exit = threading.Event()
        self._toggle_source_event = threading.Event()
        self._done_event = threading.Event()

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:

        self.get_logger().info(f"HybridTeleop.insert_cable() task: {task}")
        self._task = task
        # Reset state for new execution
        self._active_source = "policy"
        self._policy_thread_result = None
        self._teleop_loop_should_exit.clear()
        self._toggle_source_event.clear()
        self._done_event.clear()
        try:
            try:
                self._wrapped_policy = self._wrapped_policy_class(self._parent_node)
                self.get_logger().info(
                    f"Instantiated wrapped policy: {self._wrapped_policy_class.__name__}"
                )
            except Exception as e:
                self.get_logger().error(
                    f"Error instantiating wrapped policy {self._wrapped_policy_class.__name__}: {e}"
                )
                send_feedback(f"Error instantiating wrapped policy: {e}")
                return False

            try:
                self._initialize_teleop()
                self.get_logger().info(
                    f"Initialized teleop mode: {self._teleop_mode} (ready for runtime teleop toggle)"
                )
            except Exception as e:
                self.get_logger().error(
                    f"Error initializing teleop mode {self._teleop_mode}: {e}"
                )
                send_feedback(f"Error initializing teleop mode: {e}")
                self._teleop_instance = None

            gated_move_robot = self._create_gated_move_robot_callback(
                move_robot, send_feedback
            )

            self._policy_thread = threading.Thread(
                target=self._run_wrapped_policy_thread,
                args=(task, get_observation, gated_move_robot, send_feedback),
                daemon=False,
            )
            self._policy_thread.start()

            self.get_logger().info("Started wrapped policy thread")
            # Run polling loop
            teleop_exit_reason = self._run_teleop_polling_loop(
                move_robot, send_feedback
            )

            self._teleop_loop_should_exit.set()
            self._policy_thread.join(timeout=5.0)
            result = (
                self._policy_thread_result
                if self._policy_thread_result is not None
                else False
            )
            completion_reason = teleop_exit_reason or "policy_completed"

            self.get_logger().info(
                f"HybridTeleop.insert_cable() completed with result={result}, completion_reason={completion_reason}"
            )
            send_feedback(f"Task complete: {completion_reason}")

            return result

        finally:
            self._teleop_loop_should_exit.set()
            if self._teleop_instance:
                self.get_logger().info("Cleaning up teleop instance")
                if hasattr(self._teleop_instance, "disconnect"):
                    self._teleop_instance.disconnect()
                self._teleop_instance = None

    def _initialize_teleop(self):
        if self._teleop_mode == "keyboard_ee":
            from lerobot_robot_aic.aic_teleop import (
                AICKeyboardEETeleop,
                AICKeyboardEETeleopConfig,
            )

            config = AICKeyboardEETeleopConfig()
            self._teleop_instance = AICKeyboardEETeleop(config)
            self._teleop_instance.connect()
            self.get_logger().info("Keyboard EE teleop initialized and connected")
        elif self._teleop_mode == "spacemouse":
            from lerobot_robot_aic.aic_teleop import (
                AICSpaceMouseTeleop,
                AICSpaceMouseTeleopConfig,
            )

            config = AICSpaceMouseTeleopConfig()
            self._teleop_instance = AICSpaceMouseTeleop(config)
            self._teleop_instance.connect()
            self.get_logger().info("SpaceMouse teleop initialized and connected")
        else:
            raise ValueError(f"Unsupported teleop type: {self._teleop_mode}")

    def _create_gated_move_robot_callback(
        self, move_robot: MoveRobotCallback, send_feedback: SendFeedbackCallback
    ) -> MoveRobotCallback:
        def gated_move_robot(
            motion_update: MotionUpdate = None,
            joint_motion_update: JointMotionUpdate = None,
        ) -> None:
            if self._teleop_loop_should_exit.is_set():
                return

            with self._source_lock:
                if self._active_source == "policy":
                    move_robot(
                        motion_update=motion_update,
                        joint_motion_update=joint_motion_update,
                    )

        return gated_move_robot

    def _run_wrapped_policy_thread(
        self, task, get_observation, move_robot, send_feedback
    ):
        try:
            self._policy_thread_result = self._wrapped_policy.insert_cable(
                task, get_observation, move_robot, send_feedback
            )
        except Exception as e:
            self.get_logger().error(f"Exception in policy thread: {e}")
            self._policy_thread_result = False
        finally:
            self._done_event.set()

    def _run_teleop_polling_loop(self, move_robot, send_feedback) -> str:
        POLL_INTERVAL = 0.05

        self.get_logger().info(
            "Teleop polling loop started with pynput (global keyboard). Press space to toggle policy <-> teleop, enter to exit."
        )

        while not self._teleop_loop_should_exit.is_set():
            if self._done_event.is_set():
                return "policy_completed"

            # === Check for special keys from pynput ===
            if hasattr(self._teleop_instance, "misc_keys_queue"):
                try:
                    while not self._teleop_instance.misc_keys_queue.empty():
                        signal = self._teleop_instance.misc_keys_queue.get_nowait()
                        self.get_logger().info(
                            f"Teleop key signal received: {signal!r}"
                        )

                        if signal == " ":
                            with self._source_lock:
                                self._active_source = (
                                    "teleop"
                                    if self._active_source == "policy"
                                    else "policy"
                                )
                            self.get_logger().info(
                                f"TOGGLE via space: Source is now {self._active_source}"
                            )

                        elif signal in ("\r", "\n"):
                            self.get_logger().info("EXIT via enter")
                            return "user_exited"

                except queue.Empty:
                    pass
                except Exception as e:
                    self.get_logger().warning(f"Error processing keyboard queue: {e}")

            # === Get teleop action (WASD, etc.) and apply only when in teleop mode ===
            if self._teleop_instance:
                try:
                    action = self._teleop_instance.get_action()
                    with self._source_lock:
                        if self._active_source == "teleop":
                            self.get_logger().info(f"Applying teleop action: {action}")
                            move_robot(
                                motion_update=self._create_motion_update_from_teleop_action(
                                    action
                                )
                            )
                except Exception as e:
                    self.get_logger().warning(f"Error getting teleop action: {e}")

            self.sleep_for(POLL_INTERVAL)

        return "user_exited"

    def _create_motion_update_from_teleop_action(self, action) -> MotionUpdate:
        linear_x = action.get("linear.x", 0.0)
        linear_y = action.get("linear.y", 0.0)
        linear_z = action.get("linear.z", 0.0)
        angular_x = action.get("angular.x", 0.0)
        angular_y = action.get("angular.y", 0.0)
        angular_z = action.get("angular.z", 0.0)
        velocity = Twist(
            linear=Vector3(x=linear_x, y=linear_y, z=linear_z),
            angular=Vector3(x=angular_x, y=angular_y, z=angular_z),
        )
        motion_update = MotionUpdate(
            header=Header(
                frame_id="base_link",
                stamp=self._parent_node.get_clock().now().to_msg(),
            ),
            pose=Pose(),
            velocity=velocity,
            target_stiffness=np.diag([30.0, 30.0, 30.0, 20.0, 20.0, 20.0]).flatten(),
            target_damping=np.diag([50.0, 50.0, 50.0, 20.0, 20.0, 20.0]).flatten(),
            feedforward_wrench_at_tip=Wrench(
                force=Vector3(x=0.0, y=0.0, z=0.0),
                torque=Vector3(x=0.0, y=0.0, z=0.0),
            ),
            wrench_feedback_gains_at_tip=[0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_VELOCITY,
            ),
        )

        return motion_update
