"""Experimental empty-row shortcut; all insertion and visual-handoff gates retained."""
from .RunVisualConnectorServoPreloadedV37 import RunVisualConnectorServoPreloadedV37
from .RunObservedEmptyBoardTransport import RunObservedEmptyBoardTransport


class RunVisualConnectorServoEmptyRouteV38(RunVisualConnectorServoPreloadedV37):
    rule_transport_class=RunObservedEmptyBoardTransport
