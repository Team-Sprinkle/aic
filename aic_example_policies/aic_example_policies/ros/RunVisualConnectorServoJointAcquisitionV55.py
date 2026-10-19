"""Confirm translation and rotation on the same three fresh grasp observations."""
from .RunVisualConnectorServoPacketTimeV54 import RunVisualConnectorServoPacketTimeV54


class RunVisualConnectorServoJointAcquisitionV55(RunVisualConnectorServoPacketTimeV54):
    joint_grasp_acquisition = True
    # Both public SC/SFP TouchPlugins require1 s continuous contact. Preserve
    # a500 ms margin, then independently require toolkit75 and physical depth.
    seating_dwell_s = 1.5

    def _make_cached_transport(self, parent_node):
        transport = super()._make_cached_transport(parent_node)
        transport.parallel_transport_JPEG = True
        return transport
