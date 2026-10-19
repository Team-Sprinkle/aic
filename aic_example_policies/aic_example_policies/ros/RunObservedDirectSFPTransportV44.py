"""SFP high-clearance direct acquisition; no change to SC obstacle routing."""
from .RunObservedShortSFPTransportV43 import RunObservedShortSFPTransportV43


class RunObservedDirectSFPTransportV44(RunObservedShortSFPTransportV43):
    def _tcp_goal(self, port):
        goal = super()._tcp_goal(port)
        if self.family == 'sfp':
            goal[:3, 3] += port[:3, 2] * .040
        return goal

    def _build_stages(self, current, port):
        if self.family != 'sfp':
            return super()._build_stages(current, port)
        final = self._tcp_goal(port)
        height = max(float(current[2, 3]), float(final[2, 3]+.010))
        above_start = current[:3, 3].copy()
        above_start[2] = height
        above_target = final[:3, 3].copy()
        above_target[2] = height
        # Even the mating tip stays at least 80 mm before the selected opening
        # at the nominal acquisition goal. Native requested-port validation and
        # the >=120 mm measured TCP-grasp acquisition gate remain mandatory.
        self.stages = [
            ('SFP_direct_clearance', above_start, current[:3, :3].copy()),
            ('SFP_transfer_and_orient_clear', above_target, final[:3, :3].copy()),
            ('SFP_direct_visual_acquisition', final[:3, 3], final[:3, :3])]
