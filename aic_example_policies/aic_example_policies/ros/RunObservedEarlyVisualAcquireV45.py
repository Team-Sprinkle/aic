"""Acquire a sufficiently close visible SFP in place; otherwise take its route."""
import numpy as np
# The ROS parent installs the checkout's script-module import path.
from .RunObservedDirectSFPTransportV44 import RunObservedDirectSFPTransportV44
from shared_port_tcp_geometry import matrix_from_xyz_xyzw


class RunObservedEarlyVisualAcquireV45(RunObservedDirectSFPTransportV44):
    def _reset(self):
        super()._reset()
        self.early_active = False
        self.early_reference = None
        self.early_stamps = None
        self.early_fresh_count = 0
        self.early_rejections = 0
        self.initial_public_reference = None
        self.early_causal_poses = []

    def _tcp_goal(self, port):
        if self.early_active:
            return self.early_reference.copy()
        return super()._tcp_goal(port)

    def _build_stages(self, current, port):
        super()._build_stages(current, port)
        if self.family == 'sfp':
            self.fallback_stages = self.stages
            self.early_reference = (self.initial_public_reference.copy()
                                    if self.initial_public_reference is not None else current.copy())
            self.early_active = True
            self.early_started = self.last_progress
            hold = self.early_reference
            self.stages = [('SFP_initial_visual_acquisition', hold[:3,3].copy(), hold[:3,:3].copy())]

    def _setup(self, task, current, sim):
        super()._setup(task, current, sim)
        if self.early_active:
            self.reference = self.early_reference.copy()
        if self.family == 'sc' and self.empty_shortcut_selected:
            rise, transfer, orient, acquire = self.stages
            self.stages = [rise, (transfer[0], transfer[1], orient[2]), acquire]
            self._trace(event='empty_row_transfer_orientation_combined', sim_time=sim,
                        source='same_three_fresh_stationary_scene_decisions',
                        occupied_scene_route_unchanged=True, geometry_input=False)

    def _inspect(self, observation, state, task, sim):
        super()._inspect(observation, state, task, sim)
        if not self.early_active:
            return
        stamps = tuple(getattr(observation,c+'_image').header.stamp.sec*1000000000+
                       getattr(observation,c+'_image').header.stamp.nanosec
                       for c in ('center','left','right'))
        distinct = self.early_stamps is None or all(a>b for a,b in zip(stamps,self.early_stamps))
        ages = sim-np.asarray(stamps)*1e-9
        fresh = distinct and ages.min()>=-.001 and ages.max()<=.100 and max(stamps)-min(stamps)<=50000000
        packet_timing = None
        packet_mode = getattr(self,'use_packet_camera_timing',False)
        if packet_mode:
            from public_packet_camera_timing import camera_receipt
            packet_timing=camera_receipt(sim,stamps,self.early_stamps)
            fresh=bool(packet_timing['accepted'] and np.linalg.norm(state[7:10])<=.005
                       and np.linalg.norm(state[10:13])<=.03)
        if not packet_mode or fresh:
            self.early_stamps = stamps
        native = self.report['native_pose_component']
        plausible = False
        if native['accepted']:
            port = np.asarray(native['base_port_matrix'])
            tcp = np.linalg.inv(port)@matrix_from_xyz_xyzw(state[:7])
            nominal_tip = tcp[:3,3]+tcp[:3,:3]@np.asarray(self.params['body_tip_offset_m'])
            plausible = bool(.120<=-tcp[2,3]<=.250 and .070<=-nominal_tip[2]<=.150
                             and np.linalg.norm(nominal_tip[:2])<=.050)
        if packet_mode:
            if fresh:
                if plausible:
                    self.early_causal_poses=(self.early_causal_poses+[port.copy()])[-3:]
                    shift=max(np.linalg.norm(a[:3,3]-b[:3,3]) for a in self.early_causal_poses for b in self.early_causal_poses)
                    angle=max(np.arccos(np.clip((np.trace(a[:3,:3].T@b[:3,:3])-1)/2,-1.,1.))
                              for a in self.early_causal_poses for b in self.early_causal_poses)
                    if shift>.001 or angle>np.deg2rad(1.):self.early_causal_poses=[port.copy()]
                    self.early_rejections=0
                else:
                    self.early_causal_poses=[];self.early_rejections+=1
                self.early_fresh_count=len(self.early_causal_poses)
        else:
            self.early_fresh_count = self.early_fresh_count+1 if fresh and plausible else 0
            if fresh:
                self.early_rejections = self.early_rejections+1 if not plausible else 0
        self.report['valid'] = bool(self.report['valid'] and fresh and self.early_fresh_count>=3)
        self.report['reason'] = ('fresh_requested_port_in_initial_approach_region' if self.report['valid']
                                 else 'initial_acquisition_requires_fresh_views_and_public_region')
        self._trace(event='SFP_initial_acquisition_check', sim_time=sim,
                    valid=self.report['valid'], plausible_public_nominal_tip=plausible,
                    fresh_consistent_observations=self.early_fresh_count,
                    image_ages_s=ages.tolist(),packet_timing=packet_timing, geometry_input=False)

    def predict_chunk(self, observation, task_vector, *, decision_time_s=None):
        sim = self._control_sim_time(observation)
        if self.family is None:
            value = observation.controller_state.reference_tcp_pose
            p, q = value.position, value.orientation
            self.initial_public_reference = matrix_from_xyz_xyzw([p.x,p.y,p.z,q.x,q.y,q.z,q.w])
        reject_limit = getattr(self,'initial_reject_limit',0)
        repeated_rejection = reject_limit>0 and self.early_rejections>=reject_limit
        if self.early_active and (sim-self.early_started>3. or repeated_rejection) and not self.rule_final_valid:
            state = self.assembler.assemble_ros(observation)
            self.early_active = False
            self.stages = self.fallback_stages
            self.index = 0
            self.stage_streak = 0
            self.reference = matrix_from_xyz_xyzw(state[:7])
            self.compensation[:] = 0.
            self.hold = None
            self.acquisition_start = None
            self.best_error = float('inf')
            self.last_progress = sim
            self._trace(event='SFP_initial_acquisition_route_fallback', sim_time=sim,
                        reason=('repeated_fresh_invalid_initial_views' if repeated_rejection
                                else 'no_three_fresh_valid_requested_views_within3s'), geometry_input=False)
        return super().predict_chunk(observation, task_vector, decision_time_s=decision_time_s)
