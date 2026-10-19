"""Same early recovery; close a bounded tracking loop on measured waypoints."""
import numpy as np
from .RunVisualConnectorServoRecoveryV8 import RunVisualConnectorServoRecoveryV8
from .RunVisualConnectorServoRecoveryV5 import RunVisualConnectorServoRecoveryV5


class RunVisualConnectorServoRecoveryV10(RunVisualConnectorServoRecoveryV8):
    def __init__(self,parent_node):
        super().__init__(parent_node)
        self.recovery_tracking_integral=np.zeros(3);self.recovery_tracking_stamp=None

    def _retreat(self,transfer,get_observation,move_robot):
        self.recovery_tracking_integral=np.zeros(3);self.recovery_tracking_stamp=None
        return super()._retreat(transfer,get_observation,move_robot)

    def _command(self,move_robot,matrix,*,owner):
        if owner in ('measured_path_backtrack','supervisor_lateral_exploration'):
            nominal=np.asarray(matrix).copy();matrix=nominal.copy();stamp=self.local.last_observation_stamp
            state=self.retained_terminal['state'] if self.retained_terminal is not None else None
            if state is None:raise ValueError('Recovery reference needs the retained public measured state')
            dt=0. if self.recovery_tracking_stamp is None else min(.1,max(0.,stamp-self.recovery_tracking_stamp))
            self.recovery_tracking_stamp=stamp
            error=nominal[:3,3]-np.asarray(state[:3])
            self.recovery_tracking_integral=np.clip(self.recovery_tracking_integral+.6*dt*error,-.005,.005)
            matrix[:3,3]+=self.unloaded_reference_bias_base_m+self.recovery_tracking_integral
            self._trace(event='recovery_waypoint_reference_compensated',sim_time=stamp,owner=owner,
                measured_waypoint=nominal.tolist(),issued_reference=matrix.tolist(),
                bias_base_m=self.unloaded_reference_bias_base_m.tolist(),
                integral_base_m=self.recovery_tracking_integral.tolist(),measured_state=state,geometry_input=False)
        # V7 adds a constant bias. Dispatch below that override so the new
        # retained bias plus bounded integral is applied exactly once.
        RunVisualConnectorServoRecoveryV5._command(self,move_robot,matrix,owner=owner)
