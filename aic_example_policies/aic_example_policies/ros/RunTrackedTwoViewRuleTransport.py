"""Retain a causal visual landmark to place original-image refinement crops."""
import numpy as np
from scipy.spatial.transform import Rotation
from .RunTwoViewGuidedRuleTransport import RunTwoViewGuidedRuleTransport
from shared_port_tcp_geometry import matrix_from_xyz_xyzw
from build_sc_native_pose_labels import project


class RunTrackedTwoViewRuleTransport(RunTwoViewGuidedRuleTransport):
    def inspect_with_guidance(self,images,state,requested):
        original=self.strict_inspect
        if self.rule_visual_goal is not None:
            landmark=np.eye(4)
            landmark[:3,3]=self.rule_visual_goal[:3,3]-self.verifier.standoff_offset
            tcp=matrix_from_xyz_xyzw(state[:7])
            pixels=[project(landmark,tcp@self.verifier.optics[c])
                    for c in ('center','left','right')]
            if all(p[3] for p in pixels):
                self.strict_inspect=lambda im,s,r:original(im,s,r,
                    observation_candidate_pixels=[p[:2] for p in pixels])
        try:
            report=super().inspect_with_guidance(images,state,requested)
            if (self.rule_visual_goal is not None and
                    report['reason'] in ('wrong_tcp_posture','not_at_selected_standoff')):
                destination=np.asarray(report['predicted_opening_base_m'])+self.verifier.standoff_offset
                orientation=Rotation.from_quat(report['predicted_port_quat_xyzw'])*self.verifier.nominal_tcp_in_port
                nominal=np.asarray(self.rule_stages[-1][1])
                nominal_rotation=Rotation.from_matrix(self.rule_stages[-1][2])
                # A two-view reposition may exceed the old 80mm nominal
                # envelope. Permit subsequent strict three-view correction
                # within its explicitly declared 140mm guidance envelope.
                if (np.linalg.norm(destination-nominal)<=.14 and
                        np.linalg.norm(destination-self.rule_visual_goal[:3,3])<=.005 and
                        (nominal_rotation.inv()*orientation).magnitude()<=np.deg2rad(10)):
                    self.rule_visual_goal[:3,3]=destination
                    self.rule_visual_goal[:3,:3]=orientation.as_matrix()
                    self._trace(event='tracked_strict_pose_correction',sim_time=self._guidance_sim,
                                geometry_input=False,final_approval=False,
                                maximum_nominal_displacement_m=.14,maximum_rotation_deg=10)
            return report
        finally:
            self.strict_inspect=original
