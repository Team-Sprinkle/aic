"""Two-view repositioning only; strict three-view approval stays unchanged."""
from itertools import combinations
import time
import cv2
import numpy as np
import torch
from .RunRuleTransport import RunRuleTransport
from shared_port_tcp_geometry import matrix_from_xyz_xyzw
from evaluate_sc_native_triangulation import triangulate, reprojection_residual_px


class RunTwoViewGuidedRuleTransport(RunRuleTransport):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.guidance_previous=None
        self.strict_inspect=self.verifier.inspect
        self.verifier.inspect=self.inspect_with_guidance

    def inspect_with_guidance(self, images, state, requested):
        started=time.perf_counter()
        report=self.strict_inspect(images,state,requested)
        if report['valid'] or report['reason'] not in (
                'requested_identity_not_in_all_views','opening_views_disagree'):
            return report
        if 'coarse_pixels' not in report or 'spatial_identity_probability' not in report:
            return report
        cams=('center','left','right')
        coarse=np.asarray(report['coarse_pixels'])
        offsets=np.rint(coarse-80).astype(int)
        crops=[cv2.warpAffine(images[c],np.array([[1,0,-offsets[i,0]],
                [0,1,-offsets[i,1]]],np.float32),(160,160)) for i,c in enumerate(cams)]
        with torch.inference_mode():
            refined=self.verifier._refine_batch(torch.from_numpy(np.stack(crops)).permute(
                0,3,1,2).to(self.device).float()/255,requested).float().cpu().numpy().reshape(3,2)*159+offsets
        current=matrix_from_xyz_xyzw(state[:7])
        candidates=[]
        for pair in combinations(range(3),2):
            probabilities=np.asarray(report.get('effective_identity_probability',
                report['spatial_identity_probability']))[list(pair)]
            if max(probabilities)<.99 or min(probabilities)<.25:continue
            if 'candidate_association_probability' in report:
                # Presence alone must not authorize repositioning toward a
                # proposal that the explicit requested-candidate check rejects.
                association=np.asarray(report['candidate_association_probability'])[list(pair)]
                if min(association)<.9:continue
            if max(np.linalg.norm(refined[i]-coarse[i]) for i in pair)>50:continue
            points={cams[i]:refined[i] for i in pair}
            optics={c:current@self.verifier.optics[c] for c in points}
            position=triangulate(points,optics)
            residual=float(reprojection_residual_px(position,points,optics))
            destination=position+self.verifier.standoff_offset
            nominal=np.asarray(self.rule_stages[-1][1])
            if (not np.isfinite(position).all() or residual>5 or
                    abs(position[2]-.03)>.01 or np.linalg.norm(destination-nominal)>.14):continue
            candidates.append((residual,destination,pair,position))
        candidates.sort(key=lambda x:x[0])
        accepted=False
        if candidates:
            residual,destination,pair,position=candidates[0]
            if self.guidance_previous is not None and np.linalg.norm(
                    destination-self.guidance_previous)<=.001:
                if self.rule_visual_goal is None or np.linalg.norm(
                        destination-self.rule_visual_goal[:3,3])>.002:
                    goal=current.copy()
                    goal[:3,3]=destination
                    goal[:3,:3]=self.rule_stages[-1][2]
                    self.rule_visual_goal=goal
                    self.rule_last_progress_s=self._guidance_sim
                    self.rule_best_error=float('inf')
                accepted=True
            self.guidance_previous=destination
        else:
            self.guidance_previous=None
        self._trace(event='two_view_guidance',sim_time=self._guidance_sim,
                    guidance_applied=accepted,final_approval=False,geometry_input=False,
                    candidates=[{'residual_px':c[0],'tcp_goal_base_m':c[1].tolist(),
                                 'cameras':[cams[i] for i in c[2]]} for c in candidates],
                    latency_ms=(time.perf_counter()-started)*1000)
        return report

    def predict_chunk(self,observation,task_vector,**kwargs):
        self._guidance_sim=self._control_sim_time(observation)
        return super().predict_chunk(observation,task_vector,**kwargs)

    def reset_for_task(self,task):
        super().reset_for_task(task)
        self.guidance_previous=None
