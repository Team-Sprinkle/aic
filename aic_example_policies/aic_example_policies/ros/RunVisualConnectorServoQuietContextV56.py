"""Keep three quiet SC context images when shortening initial acquisition."""
from .RunVisualConnectorServoJointAcquisitionV55 import RunVisualConnectorServoJointAcquisitionV55


class RunVisualConnectorServoQuietContextV56(RunVisualConnectorServoJointAcquisitionV55):
    required_SC_far_context_frames = 3
