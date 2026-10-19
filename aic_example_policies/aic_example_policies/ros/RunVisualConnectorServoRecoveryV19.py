"""Apply frozen training-only SC pose calibration before the first contact."""
import numpy as np
from scipy.spatial.transform import Rotation
from .RunVisualConnectorServoRecoveryV18 import RunVisualConnectorServoRecoveryV18


class RunVisualConnectorServoRecoveryV19(RunVisualConnectorServoRecoveryV18):
    def _visual_servo_insert(self, task, get_observation, move_robot, send_feedback, **kwargs):
        if self.local.task_vector(task)[0] != 1 and self.angle_retry == 0:
            # No scene lookup or current-episode label: the same frozen
            # acquisition-domain calibration is used for every first attempt.
            candidate = self.calibration_proposals[0]
            self.servo_goal_rotation = Rotation.from_rotvec(
                np.deg2rad(candidate['rotation_rotvec_deg'])).as_matrix()
            self.servo_goal_translation = np.asarray(candidate['translation_m'])
            self._trace(event='training_calibration_before_first_contact',
                        rotation_rotvec_deg=candidate['rotation_rotvec_deg'],
                        translation_m=candidate['translation_m'],
                        source=candidate['source'], geometry_input=False,
                        current_episode_geometry_used=False)
        return super()._visual_servo_insert(
            task, get_observation, move_robot, send_feedback, **kwargs)
