"""Conditional empty-row route; occupied/uncertain scenes keep the bypass."""
import os
import time
import numpy as np
from .RunSampleTransportRecovery import RunSampleTransportRecovery
from .RunRuleTransport import RunRuleTransport
from empty_row_observation import EmptyRowObservation


class RunObservedEmptyBoardTransport(RunSampleTransportRecovery):
    def __init__(self,parent_node):
        super().__init__(parent_node)
        self.empty_row=EmptyRowObservation(os.environ['AIC_EMPTY_ROW_CHECKPOINT'],os.environ['AIC_EMPTY_ROW_APPROVAL'],str(self.device))

    def _reset(self):
        super()._reset()
        self.empty_row_streak=0;self.empty_row_last_stamps=None;self.empty_shortcut_selected=False

    def _inspect(self,observation,state,task,sim):
        super()._inspect(observation,state,task,sim)
        if self.family!='sc' or self.empty_shortcut_selected or self._transport_attempt!=0 or self.index>1:return
        started=time.perf_counter()
        images={c:RunRuleTransport.native(getattr(observation,c+'_image')) for c in ('center','left','right')}
        stamps=tuple(getattr(observation,c+'_image').header.stamp.sec*1000000000+getattr(observation,c+'_image').header.stamp.nanosec for c in ('center','left','right'))
        result=self.empty_row.predict(images)
        distinct=self.empty_row_last_stamps is None or all(x>y for x,y in zip(stamps,self.empty_row_last_stamps))
        fresh=distinct and max(sim-s/1e9 for s in stamps)<=.1 and min(sim-s/1e9 for s in stamps)>=-.001 and max(stamps)-min(stamps)<=50000000
        self.empty_row_last_stamps=stamps
        self.empty_row_streak=self.empty_row_streak+1 if result['clear'] and fresh else 0
        self._trace(event='empty_row_visual_check',sim_time=sim,**result,consistent_observations=self.empty_row_streak,geometry_input=False,elapsed_ms=(time.perf_counter()-started)*1000)
        if self.empty_row_streak<3:return
        # Prior supports a transport waypoint only. The final hold still
        # requires the original requested-port visual acquisition gate.
        port=self.pose.copy() if self.pose is not None else self.nominal_port.copy()
        goal=self._tcp_goal(port);current=state[:3].copy();height=max(float(current[2]),float(goal[2,3]+.04))
        if height>.55 or np.linalg.norm(goal[:3,3]-current)>.6:return
        rotation=self.reference[:3,:3].copy()
        self.stages=[('empty_row_rise_clearance',np.array([current[0],current[1],height]),rotation),
                     ('empty_row_transfer_clearance',np.array([goal[0,3],goal[1,3],height]),rotation),
                     ('empty_row_orient_above_target',np.array([goal[0,3],goal[1,3],height]),goal[:3,:3]),
                     ('empty_row_visual_acquisition',goal[:3,3],goal[:3,:3])]
        self.index=0;self.stage_streak=0;self.last_progress=sim;self.best_error=float('inf');self.acquisition_start=None
        self.empty_shortcut_selected=True
        self._trace(event='empty_row_shortcut_selected',sim_time=sim,source='three fresh full-image empty-row decisions',port_waypoint_source='accepted_visual_pose' if self.pose is not None else 'public_training_prior_only',final_requested_visual_gate_unchanged=True,reference_preserved=True,geometry_input=False)
