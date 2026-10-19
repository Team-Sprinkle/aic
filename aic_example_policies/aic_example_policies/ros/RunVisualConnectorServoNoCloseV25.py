"""Matched V21 ablation: preserve every controller change except close update."""
from .RunVisualConnectorServoRecoveryV21 import RunVisualConnectorServoRecoveryV21


class RunVisualConnectorServoNoCloseV25(RunVisualConnectorServoRecoveryV21):
    def _visual_servo_insert(self, task, get_observation, move_robot, send_feedback, **kwargs):
        if self.local.task_vector(task)[0] != 1 and self.close_update_required:
            self._trace(event='close_grasp_update_disabled_for_matched_ablation',
                        geometry_input=False, all_other_V21_controls_unchanged=True)
            self.close_update_required = False
        return super()._visual_servo_insert(task, get_observation, move_robot, send_feedback, **kwargs)
