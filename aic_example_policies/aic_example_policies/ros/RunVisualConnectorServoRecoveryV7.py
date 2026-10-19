"""Early recovery with public free-space tracking compensation for measured paths.

Historical measured waypoints are physical goals, whereas the impedance API
expects a command reference. No true plug/port position or completion flag is
used to bridge that distinction.
"""
import numpy as np
from .RunVisualConnectorServoRecoveryV6 import RunVisualConnectorServoRecoveryV6


class RunVisualConnectorServoRecoveryV7(RunVisualConnectorServoRecoveryV6):
    def __init__(self,parent_node):
        super().__init__(parent_node)
        self.unloaded_reference_bias_base_m=np.zeros(3)

    def _lock(self,task_vector,get_observation,move_robot,*,fresh_context):
        accepted=super()._lock(task_vector,get_observation,move_robot,fresh_context=fresh_context)
        if accepted:
            end=self.approach_measured_history[-1][0]
            free=[state[13:16] for stamp,state in self.approach_measured_history
                  if end-stamp<=.5 and self._force(state)<3. and np.linalg.norm(state[7:10])<.005]
            if not free:raise ValueError('No unloaded causal tracking-error samples for measured recovery')
            bias=np.median(free,axis=0)
            if not np.isfinite(bias).all() or np.linalg.norm(bias)>.02:raise ValueError('Invalid free-space tracking compensation')
            self.unloaded_reference_bias_base_m=bias
            self._trace(event='free_tracking_bias_retained',sim_time=end,bias_base_m=bias.tolist(),
                source='public controller tcp_error at free25mm approach; same controller gains',geometry_input=False)
        return accepted

    def _command(self,move_robot,matrix,*,owner):
        if owner in ('measured_path_backtrack','supervisor_lateral_exploration'):
            nominal=np.asarray(matrix).copy();matrix=nominal.copy()
            matrix[:3,3]+=self.unloaded_reference_bias_base_m
            state=self.retained_terminal
            self._trace(event='recovery_waypoint_reference_compensated',sim_time=self.local.last_observation_stamp,
                owner=owner,measured_waypoint=nominal.tolist(),issued_reference=matrix.tolist(),
                bias_base_m=self.unloaded_reference_bias_base_m.tolist(),
                measured_state=(state['state'] if state is not None else None),geometry_input=False)
        super()._command(move_robot,matrix,owner=owner)
