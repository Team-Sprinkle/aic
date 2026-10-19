"""Move to a useful view after three fresh rejections; no blind descent."""
import os
import numpy as np
from evaluate_sc_native_triangulation import FOCAL,triangulate
from shared_port_tcp_geometry import matrix_from_xyz_xyzw
from .RunObservedEarlyVisualAcquireV45 import RunObservedEarlyVisualAcquireV45
from .RunObservedDirectSFPTransportV44 import RunObservedDirectSFPTransportV44


class RunObservedEarlyVisualAcquireV46(RunObservedEarlyVisualAcquireV45):
    initial_reject_limit = 3

    def _reset(self):
        super()._reset()
        self.coarse_candidates=[]
        self.coarse_stable=None
        self.coarse_fallback_applied=False

    def _fallback_from_port(self,current,port):
        final=RunObservedDirectSFPTransportV44._tcp_goal(self,port)
        above=final[:3,3].copy()
        above[2]=max(float(current[2,3]),float(above[2]))
        lift=float(os.environ.get('AIC_SFP_FALLBACK_LIFT_M','0'))
        if not 0. <= lift <= .10:raise ValueError('SFP fallback lift must be in [0, 0.10] m')
        before=[]
        if lift:
            lifted=current[:3,3].copy();lifted[2]+=lift
            above[2]=max(above[2],lifted[2])
            before=[('SFP_lift_before_transfer',lifted,current[:3,:3])]
        return [*before,
            ('SFP_transfer_orientation_without_blind_descent',above,final[:3,:3]),
            ('SFP_requested_native_visual_acquisition',final[:3,3],final[:3,:3])]

    def _build_stages(self, current, port):
        super()._build_stages(current, port)
        if self.family != 'sfp':
            return
        self.fallback_stages=self._fallback_from_port(current,port)

    def _inspect(self,observation,state,task,sim):
        super()._inspect(observation,state,task,sim)
        if self.family!='sfp' or not self.early_active:return
        result=self.report['native_pose_component']
        pixels=result.get('coarse_pixels')
        if pixels is None or min(result.get('projected_inframe_probability',[0]))<.95:
            self.coarse_candidates=[];return
        names=('center','left','right')
        cameras={c:matrix_from_xyz_xyzw(state[:7])@self.component.optics[c] for c in names}
        try:
            point=np.asarray(triangulate(dict(zip(names,pixels)),cameras),float)
            residual=[]
            for c,pixel in zip(names,pixels):
                xyz=np.linalg.inv(cameras[c])@np.r_[point,1.]
                projected=xyz[:2]/xyz[2]*FOCAL+[576.,512.]
                residual.append(float(np.linalg.norm(projected-np.asarray(pixel))))
            nominal=self.nominal_port[:3,3]
            plausible=(np.isfinite(point).all() and max(residual)<=14.
                and abs(point[0]-nominal[0])<=.08
                and abs(point[1]-nominal[1])<=.35
                and abs(point[2]-nominal[2])<=.05)
        except (ValueError,np.linalg.LinAlgError,ZeroDivisionError):
            plausible=False;residual=[];point=None
        if not plausible:
            self.coarse_candidates=[];return
        if self.coarse_candidates and (sim-self.coarse_candidates[-1][0]>.8 or
            np.linalg.norm(point-self.coarse_candidates[-1][1])>.015):
            self.coarse_candidates=[]
        self.coarse_candidates=(self.coarse_candidates+[(sim,point)])[-3:]
        if len(self.coarse_candidates)>=2:
            self.coarse_stable=np.mean([x[1] for x in self.coarse_candidates],axis=0)
            self._trace(event='SFP_consistent_coarse_waypoint',sim_time=sim,
                predicted_opening_base_m=self.coarse_stable.tolist(),
                reprojection_px=residual,observations=len(self.coarse_candidates),
                role='transport_waypoint_only',geometry_input=False)

    def predict_chunk(self,observation,task_vector,*,decision_time_s=None):
        # The V52 latency variant shortens this to one rejection. A coarse
        # three-view waypoint needs two independent images before it may
        # steer transport, so keep a bounded three-view SFP window.
        if self.family=='sfp' and self.early_active:
            self.initial_reject_limit=3
        if self.early_active and self.coarse_stable is not None and not self.coarse_fallback_applied:
            state=self.assembler.assemble_ros(observation)
            port=self.nominal_port.copy();port[:3,3]=self.coarse_stable
            self.fallback_stages=self._fallback_from_port(matrix_from_xyz_xyzw(state[:7]),port)
            self.coarse_fallback_applied=True
            self._trace(event='SFP_coarse_guided_safe_fallback',
                predicted_opening_base_m=self.coarse_stable.tolist(),
                true_port_geometry_used=False,geometry_input=False)
        return super().predict_chunk(observation,task_vector,decision_time_s=decision_time_s)
