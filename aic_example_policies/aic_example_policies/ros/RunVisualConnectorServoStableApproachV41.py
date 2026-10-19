"""Prevent moving-reference integral windup before the 25mm observation hold."""
from .RunVisualConnectorServoWindowV40 import RunVisualConnectorServoWindowV40


class RunVisualConnectorServoStableApproachV41(RunVisualConnectorServoWindowV40):
    approach_antiwindup=True
