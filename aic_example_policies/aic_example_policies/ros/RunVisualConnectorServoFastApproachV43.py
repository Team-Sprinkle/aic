"""Bounded final speed/profile test with the same perception and physical gates."""
import numpy as np
from .RunVisualConnectorServoSFPStableV42 import RunVisualConnectorServoSFPStableV42
from .RunVisualConnectorServoRecoveryFastV34 import RunVisualConnectorServoRecoveryFastV34
from .RunObservedShortSFPTransportV43 import RunObservedShortSFPTransportV43
from requested_native_port_pose import RequestedNativePortPose


class RunVisualConnectorServoFastApproachV43(RunVisualConnectorServoSFPStableV42):
    def __init__(self, parent_node):
        RunVisualConnectorServoRecoveryFastV34.__init__(self, parent_node)
        self.cached_transport = self._make_cached_transport(parent_node)
        bundle = self.cached_transport.spec['bundle']
        for family in ('sfp', 'sc'):
            self.cached_transport.components[family] = RequestedNativePortPose(
                bundle['backbone'], bundle[family+'_coarse'],
                bundle[family+'_native_refiner'], bundle['mount_priors'],
                family=family, device=str(self.local.device), maximum_shift_px=64.,
                maximum_reprojection_px=2.5 if family == 'sc' else 2.,
                minimum_projection_probability=.9 if family == 'sc' else .95)
        self.scene_task_code = None
        self.initial_hold_reference = None

    def _make_cached_transport(self, parent_node):
        return RunObservedShortSFPTransportV43(parent_node)

    def _approach_speed(self, axial):
        blend = float(np.clip((-axial-.030)/.050, 0., 1.))
        blend = blend*blend*(3.-2.*blend)
        return .040+.040*blend

    def _lock(self, task_vector, get_observation, move_robot, *, fresh_context):
        original = list(self.local.cartesian_stiffness)
        if task_vector[0] == 1:
            self.local.cartesian_stiffness = [500., 500., 90., 50., 50., 50.]
        try:
            return super()._lock(task_vector, get_observation, move_robot,
                                 fresh_context=fresh_context)
        finally:
            self.local.cartesian_stiffness = original

    def _close_pose_servo(self, task, get_observation, move_robot, send_feedback, **kwargs):
        # SC's near-contact speed stays at the existing 12 mm/s. Only SFP's
        # longer free insertion segment tests 24 mm/s, below the 25 mm/s governor.
        self.insertion_axial_speed_m_s = .024 if self.local.task_vector(task)[0] == 1 else .012
        return super()._close_pose_servo(
            task, get_observation, move_robot, send_feedback, **kwargs)
