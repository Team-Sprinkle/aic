"""A shorter diagonal acquisition only after visual confirmation of an empty row."""
import numpy as np
from .RunObservedEarlyVisualAcquireV46 import RunObservedEarlyVisualAcquireV46


class RunObservedDirectEmptySCV49(RunObservedEarlyVisualAcquireV46):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.base_control = dict(self.spec['control'])

    def _reset(self):
        super()._reset()
        self.direct_empty_selected = False
        if hasattr(self,'base_control'):
            self.spec['control'].clear()
            self.spec['control'].update(self.base_control)

    def _tcp_goal(self, port):
        goal = super()._tcp_goal(port)
        if self.family=='sc' and self.direct_empty_selected:
            goal[:3,3] += port[:3,2]*.030
        return goal

    def _setup(self, task, current, sim):
        super()._setup(task,current,sim)
        if self.family!='sc' or not self.empty_shortcut_selected:
            return
        candidate = super()._tcp_goal(self.nominal_port)
        candidate[:3,3] += self.nominal_port[:3,2]*.030
        if min(float(current[2,3]),float(candidate[2,3]))<.140 or np.linalg.norm(candidate[:3,3]-current[:3,3])>.6:
            return
        self.direct_empty_selected = True
        self.spec['control'].update(speed_m_s=.120,angular_speed_rad_s=.8)
        self.stages = [('SC_empty_diagonal_visual_acquisition',candidate[:3,3],candidate[:3,:3])]
        self._trace(event='SC_empty_diagonal_acquisition_selected',sim_time=sim,
                    source='three_fresh_stationary_full_RGB_empty_row_decisions',
                    nominal_tip_standoff_mm=120,minimum_nominal_TCP_height_m=.140,
                    occupied_or_uncertain_route_unchanged=True,
                    native_requested_acquisition_still_required=True,geometry_input=False)
