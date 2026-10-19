"""Remove the legacy BC stand-off detour for the explicit SFP controller."""
from .RunVisualConnectorServoFastApproachV43 import RunVisualConnectorServoFastApproachV43
from .RunObservedDirectSFPTransportV44 import RunObservedDirectSFPTransportV44


class RunVisualConnectorServoDirectSFPV44(RunVisualConnectorServoFastApproachV43):
    SFP_minimum_shared_tip_standoff_m = .070

    def _make_cached_transport(self, parent_node):
        return RunObservedDirectSFPTransportV44(parent_node)
