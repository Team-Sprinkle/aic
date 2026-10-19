"""SC-only route preflight; requires no unqualified new SFP estimator."""
import numpy as np
from .RunVisualConnectorServoRecoveryFastV34 import RunVisualConnectorServoRecoveryFastV34
from .RunObservedEmptyBoardTransport import RunObservedEmptyBoardTransport
from requested_native_port_pose import RequestedNativePortPose


class RunVisualConnectorServoSCEmptyDiagnosticV38(RunVisualConnectorServoRecoveryFastV34):
    insertion_axial_speed_m_s=.012

    def __init__(self,parent_node):
        super().__init__(parent_node)
        self.cached_transport=RunObservedEmptyBoardTransport(self.parent)
        bundle=self.cached_transport.spec['bundle']
        for family in ('sfp','sc'):
            self.cached_transport.components[family]=RequestedNativePortPose(
                bundle['backbone'],bundle[family+'_coarse'],bundle[family+'_native_refiner'],
                bundle['mount_priors'],family=family,device=str(self.local.device),
                maximum_shift_px=64.,maximum_reprojection_px=2.5 if family=='sc' else 2.,
                minimum_projection_probability=.9 if family=='sc' else .95)

    def _make_rule_transport(self):return self.cached_transport

    def _approach_speed(self,axial):
        blend=float(np.clip((-axial-.030)/.050,0.,1.));blend=blend*blend*(3.-2.*blend)
        return .020+.030*blend

    def _execute(self,task,get_observation,move_robot,send_feedback,**kwargs):
        if self.local.task_vector(task)[1]!=1:raise ValueError('This preflight is SC only')
        return super()._execute(task,get_observation,move_robot,send_feedback,**kwargs)
