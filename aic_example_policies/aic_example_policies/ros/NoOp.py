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

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task


class NoOp(Policy):
    """A passive policy that keeps the task alive without commanding the robot.

    This is useful when collecting teleoperation data while the engine controls
    trial sequencing.
    """

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info(
            f"NoOp policy active for task '{task.id}'. Waiting for cancel/timeout..."
        )

        feedback_counter = 0
        while True:
            goal_handle = getattr(self._parent_node, "goal_handle", None)
            if goal_handle is None or not goal_handle.is_active:
                self.get_logger().info("NoOp detected inactive goal. Exiting.")
                return False

            if not getattr(self._parent_node, "is_active", False):
                self.get_logger().info("NoOp detected inactive lifecycle state. Exiting.")
                return False

            if feedback_counter % 5 == 0:
                send_feedback("NoOp policy active: teleoperation-controlled episode.")
            feedback_counter += 1
            self.sleep_for(0.2)
