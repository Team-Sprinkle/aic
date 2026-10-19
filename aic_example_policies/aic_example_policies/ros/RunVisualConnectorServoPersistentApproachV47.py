"""Keep the approach angular plan instead of restarting it at every measurement."""
from .RunVisualConnectorServoAcquireInPlaceV46 import RunVisualConnectorServoAcquireInPlaceV46


class RunVisualConnectorServoPersistentApproachV47(RunVisualConnectorServoAcquireInPlaceV46):
    persistent_approach_orientation = True
