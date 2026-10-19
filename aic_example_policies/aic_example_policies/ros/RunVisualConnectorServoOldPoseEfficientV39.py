"""Evaluate proven route changes while preserving the existing SFP estimator.

The wider-grasp fits remain unqualified and are not loaded by this candidate.
"""
from .RunVisualConnectorServoSCEmptyDiagnosticV38 import RunVisualConnectorServoSCEmptyDiagnosticV38
from .RunVisualConnectorServoRecoveryFastV34 import RunVisualConnectorServoRecoveryFastV34


class RunVisualConnectorServoOldPoseEfficientV39(RunVisualConnectorServoSCEmptyDiagnosticV38):
    def _execute(self,task,get_observation,move_robot,send_feedback,**kwargs):
        return RunVisualConnectorServoRecoveryFastV34._execute(
            self,task,get_observation,move_robot,send_feedback,**kwargs)
