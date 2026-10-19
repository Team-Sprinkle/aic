"""Consume a fresh stationary scene check before moving out of that view."""
import numpy as np
from .RunObservedEmptyBoardTransport import RunObservedEmptyBoardTransport
from empty_scene_receipt import receipt_matches
from .RunSampleConservativeTransport import RunSampleConservativeTransport


class RunObservedEmptyBoardWindowV40(RunObservedEmptyBoardTransport):
    def __init__(self,parent_node):
        super().__init__(parent_node);self.pending_initial_receipt=None

    def _inspect(self,observation,state,task,sim):
        # Route selection is already complete; do not spend time trying to
        # classify a row after moving away from its informative initial view.
        return RunSampleConservativeTransport._inspect(self,observation,state,task,sim)

    def _setup(self,task,current,sim):
        super()._setup(task,current,sim)
        receipt=self.pending_initial_receipt;self.pending_initial_receipt=None
        if self.family!='sc' or self._transport_attempt!=0 or not receipt_matches(receipt,task,sim,current):return
        port=self.nominal_port.copy();goal=self._tcp_goal(port)
        height=max(float(current[2,3]),float(goal[2,3]+.04))
        if height>.55 or np.linalg.norm(goal[:3,3]-current[:3,3])>.6:return
        xyz=current[:3,3];rotation=self.reference[:3,:3].copy()
        self.stages=[('empty_row_rise_clearance',np.array([xyz[0],xyz[1],height]),rotation),
                     ('empty_row_transfer_clearance',np.array([goal[0,3],goal[1,3],height]),rotation),
                     ('empty_row_orient_above_target',np.array([goal[0,3],goal[1,3],height]),goal[:3,:3]),
                     ('empty_row_visual_acquisition',goal[:3,3],goal[:3,:3])]
        self.empty_shortcut_selected=True
        self._trace(event='empty_row_shortcut_selected',sim_time=sim,source='three fresh stationary pre-route image decisions',port_waypoint_source='public_training_prior_only',final_requested_visual_gate_unchanged=True,reference_preserved=True,receipt=receipt,geometry_input=False)
