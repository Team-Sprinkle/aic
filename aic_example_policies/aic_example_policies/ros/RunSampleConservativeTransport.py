"""Sample-derived conservative transport, requested RGB acquisition, no oracle.

SC never crosses above the NIC row. Both families inspect early and use
bounded viewpoint search before fresh stationary handoff. No learned actions.
"""
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch

from .RunSharedPortTCP import RunSharedPortTCP
from .RunRuleTransport import RunRuleTransport
from requested_native_port_pose import RequestedNativePortPose
from shared_port_tcp_geometry import matrix_from_xyz_xyzw


class RunSampleConservativeTransport(RunSharedPortTCP):
    def __init__(self,parent_node):
        torch.set_num_threads(2);cv2.setNumThreads(1)
        super().__init__(parent_node)
        self.spec=json.loads(Path(os.environ['AIC_RULE_TRANSPORT_TEMPLATE']).read_text())
        if self.spec.get('schema')!='sample_conservative_transport/v1' or self.spec['geometry_runtime_input']:
            raise ValueError('Pinned public-sample specification required')
        self.target_frame='body';self.execute_steps=4
        self.transport_terminal_reason='sample_rule_acquisition_or_guarded_failure'
        self.components={};self._reset()

    def _reset(self):
        self.family=None;self.params=None;self.component=None;self.stages=None;self.index=0
        self.reference=None;self.compensation=np.zeros(3);self.last_inspection=-float('inf')
        self.pending=None;self.pose=None;self.pose_time=-float('inf');self.pose_verified_time=-float('inf');self.streak=0
        self.rule_failure=None;self.rule_final_valid=False;self.report=None;self.hold=None
        self.last_progress=None;self.best_error=float('inf');self.stage_streak=0
        self.acquisition_start=None;self.scan_index=0;self.scan_since=None
        self.marker_pending=None;self.marker_count=0;self.marker_routed=False;self.marker_started=None

    def reset_for_task(self,task):
        super().reset_for_task(task);self._reset()
        for c in self.components.values():c.reset()

    def _setup(self,task,current,sim):
        self.family='sfp' if task[0]>.5 else 'sc';self.params=self.spec['families'][self.family]
        b=self.spec['bundle']
        if self.family not in self.components:
            self.components[self.family]=RequestedNativePortPose(b['backbone'],b[self.family+'_coarse'],
                b[self.family+'_native_refiner'],b['mount_priors'],family=self.family,device=str(self.device),
                maximum_shift_px=64.,maximum_reprojection_px=2.5 if self.family=='sc' else 2.,
                minimum_projection_probability=.9 if self.family=='sc' else .95)
        self.component=self.components[self.family]
        port=int(np.argmax(task[2:4]));card=int(np.argmax(task[4:9])) if self.family=='sfp' else None
        self.requested_port=port;self.requested_card=card
        self.nominal_port=np.asarray(self.params['nominal_ports_base'][str(port) if card is None else f'{card}:{port}'])
        self.reference=current.copy();self.last_progress=sim
        self._build_stages(current,self.nominal_port)

    def _tcp_goal(self,port):
        goal=np.eye(4)
        goal[:3,:3]=port[:3,:3]@Rotation.from_quat(self.params['nominal_tcp_in_port_xyzw']).as_matrix()
        if self.family=='sc':goal[:3,3]=port[:3,3]+self.params['sc_opening_to_tcp_base_m']
        else:goal[:3,3]=port[:3,3]+port[:3,:3]@np.array([0.,0.,-.150])-goal[:3,:3]@self.params['body_tip_offset_m']
        return goal

    def _build_stages(self,current,port):
        final=self._tcp_goal(port);board=np.asarray(self.params['nominal_board_base'])
        # All original bypass waypoints come from sample board+asset row bounds.
        # A stable visual correction can move the final acquisition goal;
        # it never authorizes a blind shortcut through the row.
        if self.family=='sc':
            p=self.params['bypass'];initial=current[:3,:3]
            self.route_initial_rotation=initial.copy()
            def point(x,y,z):
                v=board@np.array([x,y,0.,1.]);v[2]=z;return v[:3]
            start_local=np.linalg.inv(board)@np.r_[current[:3,3],1.]
            final_local=np.linalg.inv(board)@np.r_[final[:3,3],1.]
            behind=p['behind_board_y_m'];beyond=p['beyond_board_y_m'];outside=p['outside_board_x_m'];low=p['low_tcp_z_m']
            route=os.environ.get('AIC_SC_TRANSPORT_ROUTE','sc_old')
            if route=='sc_old':
                entry=[('clear_start_side',point(start_local[0],behind,max(low+.05,current[2,3]-.03)),initial),
                    ('shift_behind_row_low',point(outside,behind,low),initial)]
            elif route=='sc_skip_back':
                # Move sideways from the actual start plane, with the same
                # low corridor target. Do not first draw the cable backward
                # between cards on starts that are already beside the row.
                entry=[('shift_from_start_low',point(outside,start_local[1],low),initial)]
            elif route=='sc_level_shift':
                # Clear the row sideways at the observed start height,
                # then descend only after the gripper is outside it.
                entry=[('shift_from_start_level',point(outside,start_local[1],float(current[2,3])),initial),
                    ('descend_outside_row',point(outside,start_local[1],low),initial)]
            elif route=='sc_hover':
                # Keep the plug above card tops while traversing an uncertain
                # board yaw. The final visual-acquisition leg alone descends.
                entry=[('shift_from_start_level',point(outside,start_local[1],float(current[2,3])),initial)]
            elif route=='sc_marker_scout':
                # Look toward the board at safe height before any downward
                # motion; two consistent marker fits trigger a new route.
                look=initial@Rotation.from_rotvec([.25,-.40,.08]).as_matrix()
                entry=[('shift_from_start_level',point(outside,start_local[1],float(current[2,3])),initial),
                    ('look_for_board_marker',point(outside,start_local[1],float(current[2,3])),look)]
            elif route=='sc_high_shift':
                # A cable draped against a card can be pulled into its edge
                # when the sideways leg descends. Clear it vertically, shift
                # outside the row at height, then descend away from the cards.
                raised=max(current[2,3]+.05,low+.12)
                entry=[('lift_before_side',point(start_local[0],start_local[1],raised),initial),
                    ('shift_from_start_high',point(outside,start_local[1],raised),initial),
                    ('descend_outside_row',point(outside,start_local[1],low),initial)]
            elif route=='sc_ascend':
                # First lift at the initial horizontal location, then move
                # backward at height. Descend while moving sideways only
                # after reaching the behind-row plane.
                raised=current[2,3]+.10
                entry=[('ascend_before_back',point(start_local[0],start_local[1],raised),initial),
                    ('clear_start_side_high',point(start_local[0],behind,raised),initial),
                    ('shift_behind_row_low',point(outside,behind,low),initial)]
            else:raise ValueError(f'Unknown SC transport route: {route}')
            self._trace(event='sc_transport_route_selected',route=route,
                start_board_local_m=start_local[:3].tolist(),geometry_input=False)
            early_tilt=os.environ.get('AIC_SC_EARLY_TILT','0')=='1'
            travel_z=float(current[2,3]) if route in ('sc_hover','sc_marker_scout') else low
            if early_tilt:
                finish=[('tilt_outside_row',point(outside,beyond,travel_z),final[:3,:3]),
                    ('sweep_after_row',point(final_local[0],beyond,travel_z),final[:3,:3])]
            else:
                finish=[('sweep_after_row',point(final_local[0],beyond,travel_z),initial),
                    ('tilt_clear_of_cards',point(final_local[0],beyond,travel_z),final[:3,:3])]
            self._trace(event='sc_transport_tilt_order_selected',early_tilt=early_tilt,geometry_input=False)
            self.stages=entry+[
                ('traverse_beyond_row',point(outside,beyond,travel_z),initial),
                *finish,
                ('visual_acquisition',final[:3,3],final[:3,:3])]
        else:
            z=max(current[2,3],final[2,3]+.05)
            a=current[:3,3].copy();a[2]=z
            b=final[:3,3].copy();b[2]=z
            self.stages=[('lift_clear_of_cards',a,current[:3,:3]),('transfer_above_row',b,current[:3,:3]),
                ('orient_clear_of_cards',b,final[:3,:3]),('visual_acquisition',final[:3,3],final[:3,:3])]

    def _inspect(self,observation,state,task,sim):
        images={c:RunRuleTransport.native(getattr(observation,c+'_image')) for c in ['center','left','right']}
        # Match the documented native training codec once per observation.
        if getattr(self,'parallel_transport_JPEG',False):
            from parallel_training_jpeg import normalise_triplet
            images=normalise_triplet(images)
        else:
            for c,image in images.items():
                ok,jpg=cv2.imencode('.jpg',image[...,::-1],[cv2.IMWRITE_JPEG_QUALITY,86])
                if not ok:raise RuntimeError('Native RGB codec failure')
                images[c]=cv2.cvtColor(cv2.imdecode(jpg,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        result=self.component.predict(images,state,task)
        candidate=np.asarray(result['base_port_matrix']) if result['accepted'] else None
        reason=result['reason'];valid=False
        if candidate is not None:
            # Broad sample-derived workspace gate, not a memorized per-task box.
            nominal=self.nominal_port[:3,3]
            # The released board placement can move an SFP port more than
            # 0.20 m from this sample-derived nominal. Three-view pose fit,
            # requested identity and temporal agreement remain mandatory.
            if np.linalg.norm(candidate[:3,3]-nominal)>(.35 if self.family=='sfp' else .20):
                candidate=None;reason='outside_public_sample_search_envelope'
        if candidate is not None:
            if self.pose is not None and (np.linalg.norm(candidate[:3,3]-self.pose[:3,3])>.005 or
                (Rotation.from_matrix(candidate[:3,:3]).inv()*Rotation.from_matrix(self.pose[:3,:3])).magnitude()>np.deg2rad(5.)):
                self.pose=None;self.pose_verified_time=-float('inf')
            agrees=self.pending is not None and np.linalg.norm(candidate[:3,3]-self.pending[:3,3])<=.001 and (
                Rotation.from_matrix(candidate[:3,:3]).inv()*Rotation.from_matrix(self.pending[:3,:3])).magnitude()<=np.deg2rad(1.)
            self.streak=self.streak+1 if agrees else 1;self.pending=candidate
            if self.streak>=2:self.pose=candidate;self.pose_time=sim
            if self.streak>=3:self.pose_verified_time=sim
            goal=self._tcp_goal(candidate)
            error=float(np.linalg.norm(goal[:3,3]-state[:3]));angle=float((Rotation.from_matrix(goal[:3,:3]).inv()*Rotation.from_quat(state[3:7])).magnitude())
            valid=bool(self.streak>=3 and error<=.015 and angle<=np.deg2rad(3.))
            reason='clear_selected_port_and_safe_standoff' if valid else 'not_at_selected_standoff'
        else:self.streak=0
        self.report=dict(valid=valid,reason=reason,requested_port=self.requested_port,requested_card=self.requested_card,
            predicted_opening_base_m=candidate[:3,3].tolist() if candidate is not None else None,
            predicted_port_quat_xyzw=Rotation.from_matrix(candidate[:3,:3]).as_quat().tolist() if candidate is not None else None,
            native_pose_component=result,runtime_true_geometry=False)
        event='SFP_rule_visual_inspection' if self.family=='sfp' else 'rule_visual_inspection'
        self._trace(event=event,sim_time=sim,report=self.report,geometry_input=False)
        if (self.family=='sc' and os.environ.get('AIC_SC_TRANSPORT_ROUTE')=='sc_marker_scout'
                and self.index==1 and not self.marker_routed):
            from observed_board_marker import estimate_board_marker
            estimate=estimate_board_marker(images,state,self.component.optics,
                self.params['nominal_board_base'],self.nominal_port)
            if estimate is not None:
                board=estimate['board']
                agrees=(self.marker_pending is not None and
                    np.linalg.norm(board[:3,3]-self.marker_pending[:3,3])<=.020 and
                    (Rotation.from_matrix(board[:3,:3]).inv()*
                     Rotation.from_matrix(self.marker_pending[:3,:3])).magnitude()<=np.deg2rad(3.))
                self.marker_count=self.marker_count+1 if agrees else 1
                self.marker_pending=board
                self._trace(event='SC_observed_board_marker_candidate',sim_time=sim,
                    board_base=board.tolist(),cameras=estimate['cameras'],
                    mean_reprojection_px=estimate['mean_reprojection_px'],
                    camera_disagreement_mm=estimate['camera_disagreement_mm'],
                    camera_yaw_disagreement_deg=estimate['camera_yaw_disagreement_deg'],
                    selected_port_prior_error_mm=estimate['selected_port_prior_error_mm'],
                    inference_ms=estimate['inference_ms'],consistent_frames=self.marker_count,
                    geometry_input=False)
                if self.marker_count>=2:self._replan_from_observed_board(board,state,sim)
            else:self.marker_count=0;self.marker_pending=None
        self.last_inspection=sim

    def _replan_from_observed_board(self,board,state,sim):
        original_board=np.asarray(self.params['nominal_board_base'])
        corrected_port=board@np.linalg.inv(original_board)@self.nominal_port
        self.nominal_port=corrected_port
        current=matrix_from_xyz_xyzw(state[:7]);final=self._tcp_goal(corrected_port)
        start=np.linalg.inv(board)@np.r_[current[:3,3],1.]
        target=np.linalg.inv(board)@np.r_[final[:3,3],1.]
        outside=-.23 if start[0]<-.07 else .15
        beyond=.16;low=self.params['bypass']['low_tcp_z_m']
        def point(x,y,z):
            v=board@np.array([x,y,0.,1.]);v[2]=z;return v[:3]
        original=self.route_initial_rotation
        self.stages=[
            ('clear_observed_card_row_side',point(outside,start[1],current[2,3]),original),
            ('descend_outside_observed_card_row',point(outside,start[1],low),original),
            ('travel_beside_observed_card_row',point(outside,beyond,low),original),
            ('tilt_beyond_observed_card_row',point(outside,beyond,low),final[:3,:3]),
            ('sweep_toward_observed_port',point(target[0],beyond,low),final[:3,:3]),
            ('visual_acquisition',final[:3,3],final[:3,:3])]
        self.index=0;self.stage_streak=0;self.reference=current.copy();self.compensation[:]=0
        self.last_progress=sim;self.best_error=float('inf');self.hold=None
        self.pose=None;self.pending=None;self.streak=0;self.pose_verified_time=-float('inf')
        self.marker_routed=True
        self._trace(event='SC_observed_board_route_selected',sim_time=sim,
            board_base=board.tolist(),corrected_requested_port_base=corrected_port.tolist(),
            start_board_local_m=start[:3].tolist(),outside_board_x_m=outside,
            selected_port_source='observed_marker_plus_static_asset_layout',
            true_scene_geometry_used=False,geometry_input=False)

    def predict_chunk(self,observation,task_vector,*,decision_time_s=None):
        tick=time.perf_counter();state=self.assembler.assemble_ros(observation);sim=self._control_sim_time(observation)
        current=matrix_from_xyz_xyzw(state[:7]);c=self.spec['control']
        if self.first_sim_time is None:self.first_sim_time=sim
        self.last_elapsed_sim_sec=sim-self.first_sim_time
        if self.family is None:self._setup(task_vector,current,sim)
        inspected=sim-self.last_inspection>=c['inspection_interval_s']
        if inspected:self._inspect(observation,state,task_vector,sim)
        name,xyz,rotation=self.stages[self.index]
        marker_wait=(self.family=='sc' and os.environ.get('AIC_SC_TRANSPORT_ROUTE')=='sc_marker_scout'
            and name=='look_for_board_marker' and not self.marker_routed)
        if marker_wait:
            if self.marker_started is None:self.marker_started=sim
            if sim-self.marker_started>6.:self.rule_failure='observed_board_marker_not_found'
        goal=np.eye(4);goal[:3,3]=xyz;goal[:3,:3]=rotation
        acquisition=self.index==len(self.stages)-1
        if acquisition:
            if self.acquisition_start is None:self.acquisition_start=sim
            # Keep driving to a recently confirmed static port during a brief
            # view dropout. Otherwise the scan can undo the final millimeters
            # of alignment and prevent the stationary handoff hold.
            verified_recent=self.pose is not None and sim-self.pose_verified_time<=1.5
            if verified_recent:
                goal=self._tcp_goal(self.pose)
            elif sim-self.acquisition_start>2.:
                # Search only around an existing conservative acquisition hold.
                # No descent toward an unverified opening and no oracle crop.
                offsets=[(0,0,0),(.01,0,0),(-.01,0,0),(0,.01,0),(0,-.01,0),(0,0,.01),(.02,0,.01),(-.02,0,.01),(0,.02,.01),(0,-.02,.01)]
                if self.scan_since is None:self.scan_since=sim
                if sim-self.scan_since>1.5:
                    self.scan_index=(self.scan_index+1)%len(offsets);self.scan_since=sim
                    self.reference=current.copy();self.compensation[:]=0
                goal[:3,3]+=np.asarray(offsets[self.scan_index])
            if sim-self.acquisition_start>c['acquisition_budget_s']:self.rule_failure='sample_visual_acquisition_budget'
        error=float(np.linalg.norm(goal[:3,3]-state[:3]));angle=float((Rotation.from_matrix(current[:3,:3]).inv()*Rotation.from_matrix(goal[:3,:3])).magnitude())
        progress=error+.10*angle
        if progress<self.best_error-.002:self.best_error=progress;self.last_progress=sim
        close=error<=c['stage_tolerance_m'] and angle<=np.deg2rad(3.)
        self.stage_streak=self.stage_streak+1 if close else 0
        if not acquisition and self.stage_streak>=3 and not marker_wait:
            self._trace(event='rule_stage_completed',sim_time=sim,stage=name,position_error_mm=error*1000)
            self.index+=1;self.stage_streak=0;self.reference=current.copy();self.compensation[:]=0
            self.best_error=float('inf');self.last_progress=sim
        if not acquisition and sim-self.last_progress>c['stage_no_progress_s']:self.rule_failure='sample_stage_no_measured_progress'
        stationary=np.linalg.norm(state[7:10])<=.005 and np.linalg.norm(state[10:13])<=np.deg2rad(3.)
        fresh=bool(acquisition and self.pose is not None and sim-self.pose_verified_time<=1.5 and
            np.linalg.norm(self._tcp_goal(self.pose)[:3,3]-state[:3])<=.015 and
            (Rotation.from_matrix(self._tcp_goal(self.pose)[:3,:3]).inv()*Rotation.from_quat(state[3:7])).magnitude()<=np.deg2rad(3.))
        if acquisition and fresh and stationary:
            if self.hold is None:self.hold=sim
            self.rule_final_valid=sim-self.hold>=c['hold_s']
        else:self.hold=None
        # Bounded integral compensates compliant-controller static tracking
        # bias, never unbounded pushing against an obstruction.
        if error<.030 and not self.rule_failure:
            self.compensation+=c['position_integral_gain']*.2*(goal[:3,3]-state[:3])
            self.compensation=np.clip(self.compensation,-c['maximum_tracking_compensation_m'],c['maximum_tracking_compensation_m'])
        goal[:3,3]+=self.compensation
        if self.rule_failure:goal=current.copy();self.reference=current.copy()
        reference=self.reference
        distance=float(np.linalg.norm(goal[:3,3]-reference[:3,3]));rot=float((Rotation.from_matrix(reference[:3,:3]).inv()*Rotation.from_matrix(goal[:3,:3])).magnitude())
        duration=max(distance/c['speed_m_s'],rot/c['angular_speed_rad_s'],.2)
        interp=Slerp([0.,1.],Rotation.from_matrix(np.stack([reference[:3,:3],goal[:3,:3]])))
        targets=[]
        for j in range(4):
            alpha=min((j+1)/self.control_hz/duration,1.);target=np.eye(4)
            target[:3,3]=reference[:3,3]+alpha*(goal[:3,3]-reference[:3,3]);target[:3,:3]=interp([alpha]).as_matrix()[0]
            relative=np.linalg.inv(current)@target;targets.append(np.r_[relative[:3,3],relative[:3,:2].T.reshape(-1)])
        self.reference=target.copy()
        self._trace(event='sample_rule_progress',sim_time=sim,stage=name,error_mm=error*1000,
            orientation_error_deg=float(np.rad2deg(angle)),failure=self.rule_failure,scan_index=self.scan_index,geometry_input=False)
        elapsed=(time.perf_counter()-tick)*1000;self._inference_latency_ms.append(elapsed)
        self._trace(event='sample_active_perception_timing',sim_time=sim,active=inspected,inference_ms=elapsed)
        return np.asarray(targets,np.float32),np.zeros(9,np.float32),state

    def _should_stop_transport(self,task_vector):return self.rule_final_valid or self.rule_failure is not None

    def _inspect_transport_handoff(self,observed,state,task_vector,get_observation,move_robot,decision_sim):
        if self.rule_final_valid and self.pose is not None:
            # This is a recent multi-frame visual consensus, not a new image
            # or a simulator pose. Preserve its age for offline scoring.
            self._trace(event='rule_cached_visual_handoff',sim_time=decision_sim,
                report=dict(valid=True,reason='recent_consensus_standoff',
                    requested_port=self.requested_port,requested_card=self.requested_card,
                    predicted_opening_base_m=self.pose[:3,3].tolist(),
                    predicted_port_quat_xyzw=Rotation.from_matrix(self.pose[:3,:3]).as_quat().tolist(),
                    visual_pose_age_s=decision_sim-self.pose_verified_time,
                    runtime_true_geometry=False),geometry_input=False)
        self._trace(event='rule_terminal',sim_time=decision_sim,state=state.tolist(),acquired=self.rule_final_valid,
            failure=self.rule_failure,report=self.report,insertion_enabled=False)
        return True
