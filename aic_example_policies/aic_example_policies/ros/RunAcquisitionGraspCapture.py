"""Perception-only SC hold at the actual visually accepted acquisition pose."""
from .RunSampleTransportRecovery import RunSampleTransportRecovery
from .RunGraspObservationCapture import RunGraspObservationCapture
import numpy as np
from scipy.spatial.transform import Rotation


class RunAcquisitionGraspCapture(RunSampleTransportRecovery):
    def predict_chunk(self, observation, task_vector, **kwargs):
        actions = super().predict_chunk(observation, task_vector, **kwargs)
        # This admits a VIEWPOINT for grasp supervision, never an insertion
        # handoff. A correct requested-port consensus is not a grasp label.
        # Native visual acceptance and proximity bound the safe free-space
        # capture; the following five-second hold supplies quiet image labels.
        if self.index == len(self.stages)-1 and self.report:
            native = self.report.get('native_pose_component', {})
            if native.get('accepted') and self.report.get('predicted_opening_base_m') is not None:
                port = np.asarray(native['base_port_matrix'])
                state = self.assembler.assemble_ros(observation)
                goal = self._tcp_goal(port)
                distance = np.linalg.norm(goal[:3, 3]-state[:3])
                angle = (Rotation.from_matrix(goal[:3, :3]).inv()*Rotation.from_quat(state[3:7])).magnitude()
                if distance <= .015 and angle <= np.deg2rad(3):
                    self.rule_final_valid = True
                    self._trace(event='perception_only_capture_view_accepted',
                                sim_time=self._control_sim_time(observation),
                                viewpoint_error_mm=float(distance*1000),
                                deployment_handoff_gate_passed=False,
                                no_insertion_attempt=True, geometry_input=False)
        return actions

    def insert_cable(self, task, get_observation, move_robot, send_feedback, **kwargs):
        super().insert_cable(task, get_observation, move_robot, send_feedback, **kwargs)
        if not self.rule_final_valid:
            self._trace(event='acquisition_capture_failed', reason=self.rule_failure,
                        geometry_input=False, BC_expert=False)
            return False
        self._trace(event='acquisition_capture_started', geometry_input=False,
                    BC_expert=False, visual_capture_view_accepted=True,
                    deployment_handoff_success_claim=False)
        return RunGraspObservationCapture.insert_cable(
            self, task, get_observation, move_robot, send_feedback, **kwargs)
