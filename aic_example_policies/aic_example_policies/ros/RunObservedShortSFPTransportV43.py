"""SFP acquisition at 120 mm; preserve SC's conditional obstacle-aware route."""
from .RunObservedEmptyBoardWindowV40 import RunObservedEmptyBoardWindowV40


class RunObservedShortSFPTransportV43(RunObservedEmptyBoardWindowV40):
    def _tcp_goal(self, port):
        goal = super()._tcp_goal(port)
        if self.family == 'sfp':
            # Still inside the existing 100–185 mm tip acquisition envelope.
            # The position estimator retains its separate >=120 mm TCP gate.
            goal[:3, 3] += port[:3, 2] * .030
        return goal
