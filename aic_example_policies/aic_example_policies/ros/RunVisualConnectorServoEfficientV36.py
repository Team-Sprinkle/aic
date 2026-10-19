"""Cycle-two speed hypothesis; geometry/force/seat gates remain unchanged."""
import numpy as np
from .RunVisualConnectorServoSFPGraspV35 import RunVisualConnectorServoSFPGraspV35


class RunVisualConnectorServoEfficientV36(RunVisualConnectorServoSFPGraspV35):
    insertion_axial_speed_m_s=.012

    def _approach_speed(self,axial):
        blend=float(np.clip((-axial-.030)/.050,0.,1.))
        blend=blend*blend*(3.-2.*blend)
        return .020+.030*blend
