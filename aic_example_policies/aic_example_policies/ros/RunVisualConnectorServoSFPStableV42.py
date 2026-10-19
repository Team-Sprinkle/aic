"""Test SFP lateral tracking stiffness without changing axial compliance."""
from .RunVisualConnectorServoStableApproachV41 import RunVisualConnectorServoStableApproachV41


class RunVisualConnectorServoSFPStableV42(RunVisualConnectorServoStableApproachV41):
    def _warm_grasp(self):
        super()._warm_grasp()
        self.far_context_prepared = False

    def _prepare_far_orientation(self):
        if self.far_context_prepared:
            return 0.
        elapsed = super()._prepare_far_orientation()
        self.far_context_prepared = True
        return elapsed

    def _approach_tcp_rotation(self, port):
        rotation = super()._approach_tcp_rotation(port)
        if self.scene_task_code[1] == 1:
            elapsed = self._prepare_far_orientation()
            self._trace(event='far_context_prepared_before_approach',
                        complete_preparation_ms=elapsed,
                        source='same_three_retained_far_RGB_observations',
                        geometry_input=False)
        return rotation

    def _close_pose_servo(self, task, get_observation, move_robot, send_feedback, **kwargs):
        if self.local.task_vector(task)[0] != 1:
            return super()._close_pose_servo(
                task, get_observation, move_robot, send_feedback, **kwargs)
        original = list(self.local.cartesian_stiffness)
        self.local.cartesian_stiffness = [500., 500., 90., 50., 50., 50.]
        self._trace(
            event='SFP_lateral_stiffness_diagnostic',
            stiffness_base=self.local.cartesian_stiffness,
            damping=self.local.cartesian_damping,
            axial_stiffness_unchanged=True, force_guards_unchanged=True,
            geometry_input=False)
        try:
            return super()._close_pose_servo(
                task, get_observation, move_robot, send_feedback, **kwargs)
        finally:
            self.local.cartesian_stiffness = original
