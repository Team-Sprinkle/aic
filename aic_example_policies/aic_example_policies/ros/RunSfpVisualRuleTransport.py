"""SFP rule acquisition at150mm; no learned transport or insertion actions.

Only public task, RGB, robot state and training-only static priors are used.
The precise live port pose comes from the requested-SFP visual pipeline.
"""
import json
import os
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
from .RunSharedPortTCP import RunSharedPortTCP
from .RunRuleTransport import RunRuleTransport
from shared_port_tcp_geometry import matrix_from_xyz_xyzw
from sfp_selected_port_observation import SelectedSfpPort


class RunSfpVisualRuleTransport(RunSharedPortTCP):
    def __init__(self,parent_node):
        torch.set_num_threads(4);cv2.setNumThreads(2)
        super().__init__(parent_node);self.target_frame='body';self.execute_steps=4
        self.transport_terminal_reason='SFP_rule_visual_acquisition_or_guarded_failure'
        self.priors=json.loads(Path(os.environ['AIC_SFP_RULE_PRIORS']).read_text())
        if self.priors['geometry_runtime_input']:raise ValueError('Current scene geometry is prohibited')
        self.observer=self._create_observer()
        self.observer.warmup();self._reset_rule()

    def _create_observer(self):
        return SelectedSfpPort(os.environ['AIC_SFP_RULE_BOARD_CHECKPOINT'],os.environ['AIC_SFP_RULE_CROP_CHECKPOINT'],str(self.device),
            native_refiner_checkpoint=os.environ['AIC_SFP_RULE_NATIVE_POSE_CHECKPOINT'],native_pose_mount_priors=os.environ['AIC_SFP_RULE_MOUNT_PRIORS'])

    def _reset_rule(self):
        self.stage=0;self.reference=None;self.initial=None;self.lift_z=None
        self.rule_failure=None;self.rule_final_valid=False;self.last_report=None
        self.last_inspection=-float('inf');self.last_candidate=None;self.visual_port=None
        self.visual_streak=0;self.hold_since=None;self.stage_streak=0;self.best_error=float('inf');self.last_progress=None

    def reset_for_task(self,task):
        super().reset_for_task(task);self._reset_rule()

    def _tcp_goal(self,port):
        goal=np.eye(4)
        goal[:3,:3]=port[:3,:3]@Rotation.from_quat(self.priors['nominal_TCP_in_port_xyzw']).as_matrix()
        goal[:3,3]=(port[:3,3]+port[:3,:3]@np.array([0.,0.,-.150])-
                       goal[:3,:3]@np.asarray(self.priors['mean_body_tip_position_m']))
        return goal

    def _update_acquisition_guidance(self,result,state,sim,prior):
        """Optional approximate observation-only guidance; cannot approve handoff."""
        return None

    def _acquisition_goal(self,nominal,visual_goal,current,sim,prior):
        goal=visual_goal.copy()
        if not getattr(self,'nominal_plane_acquisition',False):goal[2,3]=self.lift_z
        return goal

    def predict_chunk(self,observation,task_vector,*,decision_time_s=None):
        tick=time.perf_counter();state=self.assembler.assemble_ros(observation);sim=self._control_sim_time(observation)
        if task_vector[0]!=1:raise ValueError('SFP rule branch requires public SFP task')
        card=int(np.argmax(task_vector[4:9]));port=int(np.argmax(task_vector[2:4]));prior=self.priors['tasks'][f'card{card}_port{port}']
        current=matrix_from_xyz_xyzw(state[:7])
        if self.first_sim_time is None:self.first_sim_time=sim
        self.last_elapsed_sim_sec=sim-self.first_sim_time
        nominal=matrix_from_xyz_xyzw([*prior['nominal_TCP_standoff_base_m'],*prior['nominal_TCP_quat_xyzw']])
        if self.initial is None:
            self.initial=current.copy();self.reference=current.copy();self.last_progress=sim
            self.lift_z=max(float(current[2,3]),float(nominal[2,3])+.05)
        if sim-self.last_inspection>=.25:
            images={c:cv2.cvtColor(RunRuleTransport.native(getattr(observation,c+'_image')),cv2.COLOR_RGB2BGR) for c in ('center','left','right')}
            result=self.observer.predict_batch([(images,state,card,port)])[0];accepted=False;reason='no_coherent_requested_pose'
            self._update_acquisition_guidance(result,state,sim,prior)
            if result['accepted'] and result['predicted_orientation_base_xyzw'] is not None:
                point=np.asarray(result['predicted_base_m'])
                within=bool(np.all(point>=prior['potential_min_base_m']) and np.all(point<=prior['potential_max_base_m']))
                if within:
                    candidate=matrix_from_xyz_xyzw([*point,*result['predicted_orientation_base_xyzw']])
                    same=(self.last_candidate is not None and np.linalg.norm(point-self.last_candidate[:3,3])<=.001 and
                          (Rotation.from_matrix(self.last_candidate[:3,:3]).inv()*Rotation.from_matrix(candidate[:3,:3])).magnitude()<=np.deg2rad(1))
                    self.visual_streak=self.visual_streak+1 if same else 1;self.last_candidate=candidate
                    if self.visual_streak>=2:
                        self.visual_port=candidate;accepted=True;reason='coherent_requested_visual_pose'
                        if self.stage==0:self.lift_z=max(self.lift_z,float(self._tcp_goal(candidate)[2,3])+.05)
                else:reason='outside_training_prior_potential_region';self.visual_streak=0
            else:self.visual_streak=0
            if not accepted:self.hold_since=None
            self.last_report=dict(valid=accepted,reason=reason,requested_card=card,requested_port=port,
                predicted_opening_base_m=(self.visual_port[:3,3].tolist() if accepted else None),
                predicted_port_quat_xyzw=(Rotation.from_matrix(self.visual_port[:3,:3]).as_quat().tolist() if accepted else None),
                view_probabilities=result['view_probabilities'],reprojection_px=result['reprojection_px'],
                coarse_pixels=result['coarse_pixels'],refined_pixels=result['refined_pixels'],
                pose_fit=result['pose_fit'],
                anchor_reason=result.get('anchor_reason'),
                three_view_anchor_available=result.get('three_view_anchor_available'),
                runtime_true_geometry=False,static_training_prior_only=True,actor_insertion_enabled=False)
            self._trace(event='SFP_rule_visual_inspection',sim_time=sim,report=self.last_report)
            self.last_inspection=sim
        visual_goal=self._tcp_goal(self.visual_port) if self.visual_port is not None else nominal
        names=('lift_clear_of_cards','transfer_above_row','orient_above_row','visual_150mm_standoff')
        stage_used=self.stage
        goal=current.copy()
        if self.stage==0:goal=self.initial.copy();goal[2,3]=self.lift_z
        elif self.stage==1:goal=self.initial.copy();goal[:2,3]=visual_goal[:2,3];goal[2,3]=self.lift_z
        elif self.stage==2:goal=visual_goal.copy();goal[2,3]=self.lift_z
        elif self.visual_port is not None and self.last_report['valid']:goal=visual_goal.copy()
        else:
            # Explicit acquisition-plane ablation. A nominal approach does not
            # authorize handoff: the fresh visual gate below still applies.
            goal=self._acquisition_goal(nominal,visual_goal,current,sim,prior)
        error=float(np.linalg.norm(goal[:3,3]-current[:3,3]));angle=float((Rotation.from_matrix(current[:3,:3]).inv()*Rotation.from_matrix(goal[:3,:3])).magnitude())
        progress=error+.1*angle
        if progress<self.best_error-.002:self.best_error=progress;self.last_progress=sim
        close=error<=.015 and angle<=np.deg2rad(3)
        self.stage_streak=self.stage_streak+1 if close else 0
        if self.stage_streak>=3 and self.stage<3:
            self._trace(event='SFP_rule_stage_completed',sim_time=sim,stage=names[self.stage],error_mm=error*1000)
            self.stage+=1;self.stage_streak=0;self.best_error=float('inf');self.last_progress=sim;self.reference=current.copy()
        if stage_used==3 and close and self.last_report and self.last_report['valid'] and np.linalg.norm(state[7:10])<=.005 and np.linalg.norm(state[10:13])<=np.deg2rad(3):
            if self.hold_since is None:self.hold_since=sim
            self.rule_final_valid=sim-self.hold_since>=2.
        else:self.hold_since=None
        if sim-self.last_progress>12:self.rule_failure='SFP_rule_no_measured_progress'
        if self.rule_failure:goal=current.copy();self.reference=current.copy()
        reference=self.reference;distance=float(np.linalg.norm(goal[:3,3]-reference[:3,3]));rotation_distance=float((Rotation.from_matrix(reference[:3,:3]).inv()*Rotation.from_matrix(goal[:3,:3])).magnitude());duration=max(distance/.08,rotation_distance/.35,.2)
        interp=Slerp([0.,1.],Rotation.from_matrix(np.stack([reference[:3,:3],goal[:3,:3]])));targets=[]
        for j in range(4):
            alpha=min((j+1)/self.control_hz/duration,1.);target=np.eye(4)
            target[:3,3]=reference[:3,3]+alpha*(goal[:3,3]-reference[:3,3]);target[:3,:3]=interp([alpha]).as_matrix()[0]
            relative=np.linalg.inv(current)@target;targets.append(np.r_[relative[:3,3],relative[:3,:2].T.reshape(-1)])
        self.reference=target.copy()
        self._trace(event='SFP_rule_progress',sim_time=sim,stage=names[stage_used],error_mm=error*1000,orientation_error_deg=float(np.degrees(angle)),failure=self.rule_failure,geometry_input=False)
        self._inference_latency_ms.append((time.perf_counter()-tick)*1000)
        return np.asarray(targets,np.float32),np.zeros(9,np.float32),state

    def _should_stop_transport(self,task_vector):return self.rule_final_valid or self.rule_failure is not None

    def _inspect_transport_handoff(self,observed,state,task_vector,get_observation,move_robot,decision_sim):
        self._trace(event='rule_terminal',sim_time=decision_sim,state=state.tolist(),acquired=bool(self.rule_final_valid),failure=self.rule_failure,report=self.last_report,insertion_enabled=False)
        return True
