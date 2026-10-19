"""Use delivered-packet time for camera admission, with <=10 ms state skew."""
from .RunVisualConnectorServoUniqueForceV53 import RunVisualConnectorServoUniqueForceV53


class RunVisualConnectorServoPacketTimeV54(RunVisualConnectorServoUniqueForceV53):
    use_packet_camera_timing = True

    def _make_cached_transport(self, parent_node):
        transport = super()._make_cached_transport(parent_node)
        transport.use_packet_camera_timing = True
        return transport
