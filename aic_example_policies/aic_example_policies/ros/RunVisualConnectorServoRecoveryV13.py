"""Apply low-force fine-contact recovery only in the near-mouth task phase."""
from .RunVisualConnectorServoRecoveryV12 import RunVisualConnectorServoRecoveryV12


class RunVisualConnectorServoRecoveryV13(RunVisualConnectorServoRecoveryV12):
    def _supervise_servo_stall(self,state,issued,point,stamp,force,depth):
        # The generic8N supervisor still covers the whole approach. The more
        # sensitive1.5N contact detector must not interrupt free-space tilt
        # or lateral alignment10mmoutside the opening.
        if point[2,3]<-.005:
            self.servo_stall_history=[]
            return
        return super()._supervise_servo_stall(state,issued,point,stamp,force,depth)
