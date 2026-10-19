"""Explicit acquisition without needless motion, preserving the public API hold."""
import numpy as np
from scipy.spatial.transform import Rotation
from .RunVisualConnectorServoDirectSFPV44 import RunVisualConnectorServoDirectSFPV44
from .RunObservedEarlyVisualAcquireV45 import RunObservedEarlyVisualAcquireV45
from shared_port_tcp_geometry import matrix_from_xyz_xyzw


class RunVisualConnectorServoAcquireInPlaceV45(RunVisualConnectorServoDirectSFPV44):
    SFP_maximum_shared_tip_lateral_m = .060

    def _make_cached_transport(self, parent_node):
        return RunObservedEarlyVisualAcquireV45(parent_node)

    def _lock(self, task_vector, get_observation, move_robot, *, fresh_context):
        observation = get_observation()
        if observation is None:
            return False
        reference = observation.controller_state.reference_tcp_pose
        position, q = reference.position, reference.orientation
        self.fresh_public_reference = matrix_from_xyz_xyzw(
            [position.x,position.y,position.z,q.x,q.y,q.z,q.w])
        measured = matrix_from_xyz_xyzw(self.local.assembler.assemble_ros(observation)[:7])
        error = np.linalg.inv(measured)@self.fresh_public_reference
        if np.linalg.norm(error[:3,3])>.040 or Rotation.from_matrix(error[:3,:3]).magnitude()>np.deg2rad(5.):
            self.fresh_public_reference = None
            self._trace(event='fresh_public_reference_rejected', geometry_input=False)
            return False
        self._trace(event='fresh_acquisition_public_API_reference_preserved',
                    source='public_controller_reference_tcp_pose',
                    reference=self.fresh_public_reference.tolist(), geometry_input=False)
        try:
            return super()._lock(task_vector, get_observation, move_robot, fresh_context=fresh_context)
        finally:
            self.fresh_public_reference = None

    def _command(self, move_robot, matrix, *, owner):
        if owner=='fresh_visual_acquisition_hold' and getattr(self,'fresh_public_reference',None) is not None:
            matrix = self.fresh_public_reference
        return super()._command(move_robot, matrix, owner=owner)
