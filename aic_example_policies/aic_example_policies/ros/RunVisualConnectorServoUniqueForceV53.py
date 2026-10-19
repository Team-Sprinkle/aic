"""Admit each distinct force observation once; test faster empty-SC insertion."""
from .RunVisualConnectorServoResponsiveV52 import RunVisualConnectorServoResponsiveV52
from .RunVisualConnectorServoBiasSeedV31 import RunVisualConnectorServoBiasSeedV31


class RunVisualConnectorServoUniqueForceV53(RunVisualConnectorServoResponsiveV52):
    unique_initial_force_samples = True
    initial_force_minimum_quiet_span_s = .9

    def _close_pose_servo(self, task, get_observation, move_robot, send_feedback, **kwargs):
        if self.local.task_vector(task)[1]==1 and getattr(self.rule,'direct_empty_selected',False):
            self.insertion_axial_speed_m_s = .024
            return RunVisualConnectorServoBiasSeedV31._close_pose_servo(
                self,task,get_observation,move_robot,send_feedback,**kwargs)
        return super()._close_pose_servo(task,get_observation,move_robot,send_feedback,**kwargs)
