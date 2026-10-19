"""Controlled two-visible-camera pose ablation, not a promoted default.

The original three-camera input is retained. A port estimate may use two
available views with the same native-landmark 2 px reprojection check. Live
correct-identity/pose/force gates must pass before this branch is promoted.
"""
from .RunSfpNominalPlaneRuleTransport import RunSfpNominalPlaneRuleTransport


class RunSfpTwoViewRuleTransport(RunSfpNominalPlaneRuleTransport):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.observer.minimum_views = 2
