"""Track free approach firmly, preserving holding effort at both gain seams."""
import numpy as np
from .RunVisualConnectorServoDirectEmptySCV49 import RunVisualConnectorServoDirectEmptySCV49
from shared_port_tcp_geometry import matrix_from_xyz_xyzw
from cartesian_hold_gain_transfer import transfer_hold


class RunVisualConnectorServoFreeApproachV50(RunVisualConnectorServoDirectEmptySCV49):
    def _public_reference(self, observation):
        value = observation.controller_state.reference_tcp_pose
        p,q = value.position,value.orientation
        return matrix_from_xyz_xyzw([p.x,p.y,p.z,q.x,q.y,q.z,q.w])

    def _transfer_approach_gains(self, old, new, *, owner):
        observation = self.gain_get_observation()
        if observation is None:
            raise ValueError('Public observation required for gain transfer')
        state = self.local.assembler.assemble_ros(observation)
        measured = matrix_from_xyz_xyzw(state[:7])
        reference = self._public_reference(observation)
        adjusted = transfer_hold(measured,reference,old,new)
        stamp = self.local._control_sim_time(observation)
        self.local.cartesian_stiffness = list(new)
        self._command(self.gain_move_robot,adjusted,owner=owner)
        self.local.governor_seed = (adjusted.copy(),stamp)
        self._trace(event='free_approach_impedance_gain_transfer',owner=owner,sim_time=stamp,
                    old_stiffness=list(old),new_stiffness=list(new),
                    old_public_reference=reference.tolist(),new_issued_reference=adjusted.tolist(),
                    static_effort_preservation_is_model_based=True,
                    damping_unchanged=True,geometry_input=False)

    def _approach_tcp_rotation(self, port):
        rotation = super()._approach_tcp_rotation(port)
        if self.free_gain_enabled:
            self.approach_old_gains = list(self.local.cartesian_stiffness)
            new = [500.,500.,240.,100.,100.,100.]
            self._transfer_approach_gains(self.approach_old_gains,new,owner='free_approach_gain_entry')
            self.approach_active_gains = new
            self.approach_bias_from_issued_reference = True
        return rotation

    def _lock(self, task_vector, get_observation, move_robot, *, fresh_context):
        self.gain_get_observation,self.gain_move_robot = get_observation,move_robot
        self.free_gain_enabled = bool(task_vector[0]==1 or getattr(self.rule,'direct_empty_selected',False))
        self.approach_old_gains,self.approach_active_gains = None,None
        self.approach_bias_from_issued_reference = False
        try:
            return super()._lock(task_vector,get_observation,move_robot,fresh_context=fresh_context)
        finally:
            try:
                if self.approach_active_gains is not None:
                    self._transfer_approach_gains(self.approach_active_gains,self.approach_old_gains,
                                                  owner='free_approach_gain_exit')
            finally:
                self.gain_get_observation,self.gain_move_robot = None,None
                self.approach_bias_from_issued_reference = False
