"""Preload immutable networks at configuration; reset every observation history."""
from .RunVisualConnectorServoEfficientV36 import RunVisualConnectorServoEfficientV36
from .RunSampleTransportRecovery import RunSampleTransportRecovery
from requested_native_port_pose import RequestedNativePortPose


class RunVisualConnectorServoPreloadedV37(RunVisualConnectorServoEfficientV36):
    rule_transport_class=RunSampleTransportRecovery

    def __init__(self,parent_node):
        super().__init__(parent_node)
        self.cached_transport=self.rule_transport_class(self.parent)
        bundle=self.cached_transport.spec['bundle']
        for family in ('sfp','sc'):
            self.cached_transport.components[family]=RequestedNativePortPose(
                bundle['backbone'],bundle[family+'_coarse'],bundle[family+'_native_refiner'],
                bundle['mount_priors'],family=family,device=str(self.local.device),
                maximum_shift_px=64.,maximum_reprojection_px=2.5 if family=='sc' else 2.,
                minimum_projection_probability=.9 if family=='sc' else .95)
        self._trace(event='immutable_transport_models_preloaded',observations_used=False,
                    task_specific_geometry_used=False,history_reuse=False,geometry_input=False)

    def _make_rule_transport(self):
        # insert_cable resets the complete rule state and each component's
        # observation history before the next task. Reuse weights only.
        return self.cached_transport
