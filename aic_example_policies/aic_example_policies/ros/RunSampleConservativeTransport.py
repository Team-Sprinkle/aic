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
        self.pending=None;self.pose=None;self.pose_time=-float('inf');self.streak=0
        self.rule_failure=None;self.rule_final_valid=False;self.report=None;self.hold=None
        self.last_progress=None;self.best_error=float('inf');self.stage_streak=0
        self.acquisition_start=None;self.scan_index=0;self.scan_since=None

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
            if early_tilt:
                finish=[('tilt_outside_row',point(outside,beyond,low),final[:3,:3]),
                    ('sweep_after_row',point(final_local[0],beyond,low),final[:3,:3])]
            else:
                finish=[('sweep_after_row',point(final_local[0],beyond,low),initial),
                    ('tilt_clear_of_cards',point(final_local[0],beyond,low),final[:3,:3])]
            self._trace(event='sc_transport_tilt_order_selected',early_tilt=early_tilt,geometry_input=False)
            self.stages=entry+[
                ('traverse_beyond_row',point(outside,beyond,low),initial),
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
            if np.linalg.norm(candidate[:3,3]-nominal)>.20:
                candidate=None;reason='outside_public_sample_search_envelope'
        if candidate is not None:
            agrees=self.pending is not None and np.linalg.norm(candidate[:3,3]-self.pending[:3,3])<=.001 and (
                Rotation.from_matrix(candidate[:3,:3]).inv()*Rotation.from_matrix(self.pending[:3,:3])).magnitude()<=np.deg2rad(1.)
            self.streak=self.streak+1 if agrees else 1;self.pending=candidate
            if self.streak>=2:self.pose=candidate;self.pose_time=sim
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
        self.last_inspection=sim

    def predict_chunk(self,observation,task_vector,*,decision_time_s=None):
        tick=time.perf_counter();state=self.assembler.assemble_ros(observation);sim=self._control_sim_time(observation)
        current=matrix_from_xyz_xyzw(state[:7]);c=self.spec['control']
        if self.first_sim_time is None:self.first_sim_time=sim
        self.last_elapsed_sim_sec=sim-self.first_sim_time
        if self.family is None:self._setup(task_vector,current,sim)
        inspected=sim-self.last_inspection>=c['inspection_interval_s']
        if inspected:self._inspect(observation,state,task_vector,sim)
        name,xyz,rotation=self.stages[self.index]
        goal=np.eye(4);goal[:3,3]=xyz;goal[:3,:3]=rotation
        acquisition=self.index==len(self.stages)-1
        if acquisition:
            if self.acquisition_start is None:self.acquisition_start=sim
            if self.pose is not None and sim-self.pose_time<=1.:
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
        if not acquisition and self.stage_streak>=3:
            self._trace(event='rule_stage_completed',sim_time=sim,stage=name,position_error_mm=error*1000)
            self.index+=1;self.stage_streak=0;self.reference=current.copy();self.compensation[:]=0
            self.best_error=float('inf');self.last_progress=sim
        if not acquisition and sim-self.last_progress>c['stage_no_progress_s']:self.rule_failure='sample_stage_no_measured_progress'
        stationary=np.linalg.norm(state[7:10])<=.005 and np.linalg.norm(state[10:13])<=np.deg2rad(3.)
        fresh=self.report and self.report['valid'] and sim-self.last_inspection<=c['inspection_interval_s']+.1
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
        self._trace(event='rule_terminal',sim_time=decision_sim,state=state.tolist(),acquired=self.rule_final_valid,
            failure=self.rule_failure,report=self.report,insertion_enabled=False)
        return True
