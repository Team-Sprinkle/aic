"""Approximate visual centering -> three-view identity anchor -> 150 mm.

Single-view nominal-plane and loose two-view estimates guide reacquisition
only. A precise three-view anchor is mandatory before two-view handoff updates.
"""
import numpy as np
from .RunSfpVisualRuleTransport import RunSfpVisualRuleTransport
from sfp_anchored_port_observation import AnchoredSfpPort
from sfp_plane_guidance import requested_plane_guidance


class RunSfpGuidedAnchoredRuleTransport(RunSfpVisualRuleTransport):
    def __init__(self,parent_node):
        super().__init__(parent_node)
        self.observer=AnchoredSfpPort(self.observer)

    def _reset_rule(self):
        super()._reset_rule()
        self.guide_history=[];self.guide_point=None
        if hasattr(self,'observer') and hasattr(self.observer,'reset_context'):
            self.observer.reset_context()

    def _update_acquisition_guidance(self,result,state,sim,prior):
        if self.stage<3 or result.get('three_view_anchor_available'):return
        guidance=result.get('guidance_candidate')
        if guidance is None:
            guidance=requested_plane_guidance(result,state,self.observer.optics,prior)
        if guidance is None:return
        point=np.asarray(guidance['position_base_m'],float)
        if not (np.all(point>=prior['potential_min_base_m']) and
                np.all(point<=prior['potential_max_base_m'])):return
        self.guide_history=(self.guide_history+[point])[-3:]
        if len(self.guide_history)<2:return
        median=np.median(self.guide_history,axis=0)
        if max(np.linalg.norm(p-median) for p in self.guide_history)>.010:return
        if self.guide_point is not None and np.linalg.norm(median-self.guide_point)>.020:return
        changed=self.guide_point is None or np.linalg.norm(median-self.guide_point)>.010
        self.guide_point=median
        if changed:
            self.best_error=float('inf');self.last_progress=sim
        self._trace(event='SFP_rough_acquisition_guidance',sim_time=sim,
            point_base_m=median.tolist(),cameras=guidance['cameras'],
            insertion_lock_allowed=False,geometry_input=False)

    def _acquisition_goal(self,nominal,visual_goal,current,sim,prior):
        goal=visual_goal.copy();goal[2,3]=self.lift_z
        if self.visual_port is None and self.guide_point is not None:
            goal[:2,3]=nominal[:2,3]+self.guide_point[:2]-np.asarray(prior['port_mean_base_m'])[:2]
        return goal
