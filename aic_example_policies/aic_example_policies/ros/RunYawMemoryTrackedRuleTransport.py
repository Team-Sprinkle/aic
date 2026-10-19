"""Causal static-board orientation memory with current-image port approval."""
import numpy as np
from scipy.spatial.transform import Rotation
from .RunGuardedYawTrackedRuleTransport import RunGuardedYawTrackedRuleTransport


class RunYawMemoryTrackedRuleTransport(RunGuardedYawTrackedRuleTransport):
    def __init__(self,parent_node):
        super().__init__(parent_node)
        self.board_anchor=None
        self.board_candidate=None
        self.board_keyframes=[]
        self.board_requested=None
        self.raw_strict_inspect=self.strict_inspect
        self.strict_inspect=self.inspect_with_board_memory

    def reset_for_task(self,task):
        super().reset_for_task(task)
        self.board_anchor=None;self.board_candidate=None;self.board_keyframes=[];self.board_requested=None

    def inspect_with_board_memory(self,images,state,requested,**kwargs):
        if self.board_requested!=requested:
            self.board_anchor=None;self.board_candidate=None;self.board_keyframes=[];self.board_requested=requested
        history={}
        if self.board_anchor is not None:
            history={'history_opening_base_m':self.board_anchor['position'],
                     'history_port_quat_xyzw':self.board_anchor['rotation'].as_quat(),
                     'history_board_spread_deg':self.board_anchor['spread']}
        report=self.raw_strict_inspect(images,state,requested,**kwargs,**history)
        if (report.get('orientation_source')=='current_board' and
                report['reason'] in ('wrong_tcp_posture','not_at_selected_standoff',
                                     'clear_selected_port_and_safe_standoff')):
            rotation=Rotation.from_quat(report['predicted_port_quat_xyzw']);position=np.asarray(report['predicted_opening_base_m'])
            candidate=self.board_candidate
            if (candidate is not None and np.linalg.norm(position-candidate['position'])<=.001 and
                    (candidate['rotation'].inv()*rotation).magnitude()<=np.deg2rad(.25)):
                anchor={'rotation':Rotation.from_quat(np.stack([candidate['rotation'].as_quat(),rotation.as_quat()])).mean(),
                        'position':(candidate['position']+position)/2,
                        'spread':max(candidate['spread'],report['board_view_spread_deg']),
                        'tcp':np.asarray(state[:7]),'sim_time':self._guidance_sim}
                self.board_anchor=anchor
                if (not self.board_keyframes or np.linalg.norm(anchor['tcp'][:3]-self.board_keyframes[-1]['tcp'][:3])>=.020 or
                        (Rotation.from_quat(self.board_keyframes[-1]['tcp'][3:]).inv()*Rotation.from_quat(anchor['tcp'][3:])).magnitude()>=np.deg2rad(3)):
                    self.board_keyframes.append(anchor);self.board_keyframes=self.board_keyframes[-10:]
            self.board_candidate={'rotation':rotation,'position':position,'spread':report['board_view_spread_deg']}
        self._trace(event='causal_board_orientation_memory',sim_time=self._guidance_sim,
                    available=self.board_anchor is not None,keyframes=len(self.board_keyframes),
                    history_used=report.get('orientation_source')=='causal_history',
                    distance_spacing_mm=20,rotation_spacing_deg=3,
                    current_identity_and_position_required=True,geometry_input=False)
        return report
