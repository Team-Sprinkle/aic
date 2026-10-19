"""Frozen-BC diagnostic with full-grasp approach and continuous guarded references."""
import numpy as np
from scipy.spatial.transform import Rotation
from .RunFullConnectorInsertion import RunFullConnectorInsertion
from .RunHistoryChunkedInsertion import HistoryChunkedLockedActor
from shared_port_tcp_geometry import matrix_from_xyz_xyzw
from insertion_reference_governor import InsertionReferenceGovernor


class ContinuousLockedActor(HistoryChunkedLockedActor):
    def __init__(self,parent,checkpoint):
        super().__init__(parent,checkpoint)
        self.governor=None;self.governor_seed=None;self.load_change_provider=None
        self.last_observation_stamp=None

    def _control_sim_time(self,observation):
        stamp=super()._control_sim_time(observation)
        self.last_observation_stamp=stamp
        return stamp

    def _bound_transport_target(self,absolute,task_vector,state):
        absolute=super()._bound_transport_target(absolute,task_vector,state)
        measured=matrix_from_xyz_xyzw(state[:7])
        if self.governor is None:
            if self.governor_seed is None:raise ValueError('No causal approach-command handover')
            reference,stamp=self.governor_seed
            grasp=np.eye(4);grasp[:3,3]=self.visual_body_tip;grasp[:3,:3]=self.visual_body_connector_rotation
            self.governor=InsertionReferenceGovernor(self.visual_port,grasp,reference,measured,stamp)
            self._trace(event='reference_governor_seeded',sim_time=self.last_observation_stamp,
                        bias_port_m=self.governor.bias.tolist(),geometry_input=False)
        result,diagnostic=self.governor.step(absolute,measured,self.last_observation_stamp,self.load_change_provider(state))
        self._trace(event='insertion_reference_governed',sim_time=self.last_observation_stamp,
                    raw_tcp_target=np.asarray(absolute).tolist(),issued_tcp_target=result.tolist(),
                    **diagnostic,geometry_input=False)
        return result


class RunContinuousConnectorInsertion(RunFullConnectorInsertion):
    def _make_local_actor(self,checkpoint):
        actor=ContinuousLockedActor(self.parent,checkpoint)
        actor.history_provider=lambda:dict(task_id=self.history_task_id,rows=self.observation_history)
        actor.load_change_provider=lambda state:self._force(state)
        return actor

    def _approach_tcp_rotation(self,port):
        if self.locked_connector_rotation is None:raise ValueError('Full RGB grasp rotation must precede approach')
        return port[:3,:3]@self.locked_connector_rotation.T

    def _approach_speed(self,axial):
        return .020 if axial<-.050 else .008

    def _command(self,move_robot,matrix,*,owner):
        super()._command(move_robot,matrix,owner=owner)
        if owner=='visual_transport_150_to25mm':
            self.local.governor=None
            self.local.governor_seed=(np.asarray(matrix).copy(),self.local.last_observation_stamp)
