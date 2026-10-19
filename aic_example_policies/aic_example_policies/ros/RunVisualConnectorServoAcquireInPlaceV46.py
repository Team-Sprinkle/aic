"""Same visual admission; remove redundant waiting before choosing its route."""
from .RunVisualConnectorServoAcquireInPlaceV45 import RunVisualConnectorServoAcquireInPlaceV45
from .RunObservedEarlyVisualAcquireV46 import RunObservedEarlyVisualAcquireV46


class RunVisualConnectorServoAcquireInPlaceV46(RunVisualConnectorServoAcquireInPlaceV45):
    def _make_cached_transport(self, parent_node):
        return RunObservedEarlyVisualAcquireV46(parent_node)
