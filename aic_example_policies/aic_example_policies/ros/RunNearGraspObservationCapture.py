"""Observation-only hold after the existing RGB/public-state 25mm approach."""
from .RunVisualConnectorServoDiagnosticV4 import RunVisualConnectorServoDiagnosticV4
from .RunGraspObservationCapture import RunGraspObservationCapture


class RunNearGraspObservationCapture(RunVisualConnectorServoDiagnosticV4):
    def _visual_servo_insert(self, task, get_observation, move_robot, send_feedback, **kwargs):
        self._trace(event='near_grasp_capture_started', nominal_standoff_mm=25,
                    geometry_input=False, learned_actor_inference=False,
                    insertion_attempt=False, BC_expert=False)
        return RunGraspObservationCapture.insert_cable(
            self, task, get_observation, move_robot, send_feedback, **kwargs)
