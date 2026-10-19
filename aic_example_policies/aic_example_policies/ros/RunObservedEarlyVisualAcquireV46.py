"""Move to a useful view after three fresh rejections; no blind descent."""
from .RunObservedEarlyVisualAcquireV45 import RunObservedEarlyVisualAcquireV45
from .RunObservedDirectSFPTransportV44 import RunObservedDirectSFPTransportV44


class RunObservedEarlyVisualAcquireV46(RunObservedEarlyVisualAcquireV45):
    initial_reject_limit = 3

    def _build_stages(self, current, port):
        super()._build_stages(current, port)
        if self.family != 'sfp':
            return
        final = RunObservedDirectSFPTransportV44._tcp_goal(self, port)
        above = final[:3,3].copy()
        above[2] = max(float(current[2,3]),float(above[2]))
        self.fallback_stages = [
            ('SFP_transfer_orientation_without_blind_descent', above, final[:3,:3]),
            ('SFP_requested_native_visual_acquisition', final[:3,3], final[:3,:3])]
