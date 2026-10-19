"""Test conditional direct empty-row SC transport; preserve insertion control."""
from .RunVisualConnectorServoPersistentApproachV48 import RunVisualConnectorServoPersistentApproachV48
from .RunObservedDirectEmptySCV49 import RunObservedDirectEmptySCV49


class RunVisualConnectorServoDirectEmptySCV49(RunVisualConnectorServoPersistentApproachV48):
    def _make_cached_transport(self, parent_node):
        return RunObservedDirectEmptySCV49(parent_node)
