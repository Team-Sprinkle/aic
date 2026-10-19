"""Monotone forward cap and public estimated-seating termination, frozen actor."""
import numpy as np
from scipy.spatial.transform import Rotation
from .RunContinuousConnectorInsertion import RunContinuousConnectorInsertion,ContinuousLockedActor
from shared_port_tcp_geometry import matrix_from_xyz_xyzw
from insertion_reference_governor_v2 import InsertionReferenceGovernor


class EstimatedSeatingReached(Exception):
    pass


class ContinuousLockedActorV2(ContinuousLockedActor):
    def __init__(self,parent,checkpoint):
        super().__init__(parent,checkpoint);self.seated_since=None

    def insert_cable(self,task,get_observation,move_robot,send_feedback,**kwargs):
        self.seated_since=None
        try:return super().insert_cable(task,get_observation,move_robot,send_feedback,**kwargs)
        except EstimatedSeatingReached:
            self.clear_pending_actions()
            self._trace(event='local_attempt_acknowledged',attempt_completed=True,
                reason='public_estimated_seating',physical_insertion_asserted=False,scoring_input=False)
            return True

    def _bound_transport_target(self,absolute,task_vector,state):
        measured=matrix_from_xyz_xyzw(state[:7]);grasp=np.eye(4)
        grasp[:3,3]=self.visual_body_tip;grasp[:3,:3]=self.visual_body_connector_rotation
        if self.governor is None:
            if self.governor_seed is None:raise ValueError('Missing causal seed')
            reference,stamp=self.governor_seed
            self.governor=InsertionReferenceGovernor(self.visual_port,grasp,reference,measured,stamp)
            self._trace(event='reference_governor_seeded',sim_time=self.last_observation_stamp,
                        version=2,bias_port_m=self.governor.bias.tolist(),geometry_input=False)
        point=np.linalg.inv(self.visual_port)@measured@grasp
        depth=.0465 if task_vector[0]==1 else .0155
        estimated=(point[2,3]>=depth and np.linalg.norm(point[:2,3])<=.0005 and
                   Rotation.from_matrix(point[:3,:3]).magnitude()<=np.deg2rad(.75) and
                   np.linalg.norm(state[7:10])<=.002)
        self.seated_since=(self.last_observation_stamp if self.seated_since is None else self.seated_since) if estimated else None
        if self.seated_since is not None and self.last_observation_stamp-self.seated_since>=.5:
            self._trace(event='terminal_before_return',reason='public_estimated_seating',
                        sim_time=self.last_observation_stamp,state=state.tolist(),
                        estimated_connector_port_m=point[:3,3].tolist(),geometry_input=False,
                        physical_insertion_asserted=False)
            raise EstimatedSeatingReached()
        return super()._bound_transport_target(absolute,task_vector,state)


class RunContinuousConnectorInsertionV2(RunContinuousConnectorInsertion):
    def _make_local_actor(self,checkpoint):
        actor=ContinuousLockedActorV2(self.parent,checkpoint)
        actor.history_provider=lambda:dict(task_id=self.history_task_id,rows=self.observation_history)
        actor.load_change_provider=lambda state:self._force(state)
        return actor
