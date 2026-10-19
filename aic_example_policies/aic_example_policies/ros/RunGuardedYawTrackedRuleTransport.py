"""Track visual crops and reject edge-saturated board orientation estimates."""
from .RunTrackedTwoViewRuleTransport import RunTrackedTwoViewRuleTransport


class RunGuardedYawTrackedRuleTransport(RunTrackedTwoViewRuleTransport):
    def __init__(self,parent_node):
        super().__init__(parent_node)
        self.verifier.board_yaw_search_rad=.21
        self.verifier.board_yaw_prior_line_deg=12.
        self.verifier.reject_yaw_search_boundary=True
