"""Explicit RGB controller with measured-path recovery and bounded angle retries.

No actor inference or true geometry. Independent physical scoring is required;
estimated seating at an exploratory angle is not itself a success claim.
"""
import os,time
import numpy as np
from scipy.spatial.transform import Rotation
from .RunVisualConnectorServoDiagnosticV4 import RunVisualConnectorServoDiagnosticV4
from .RunRuleThenSharedRecovery import measured_pose,RecoveryTransfer
from shared_port_tcp_geometry import matrix_from_xyz_xyzw
from insertion_reference_governor_v2 import InsertionReferenceGovernor,ReferenceLimits


class RunVisualConnectorServoRecoveryV5(RunVisualConnectorServoDiagnosticV4):
    def __init__(self,parent_node):
        requested=os.environ.get('AIC_SHARED_RECOVERY_MODE','none')
        if requested not in ('none','local','hierarchical'):raise ValueError('Unknown recovery arm')
        previous=os.environ.get('AIC_SHARED_RECOVERY_MODE')
        os.environ['AIC_SHARED_RECOVERY_MODE']='none'
        try:super().__init__(parent_node)
        finally:
            if previous is None:os.environ.pop('AIC_SHARED_RECOVERY_MODE',None)
            else:os.environ['AIC_SHARED_RECOVERY_MODE']=previous
        self.requested_recovery=requested;self.mode=requested
        self.servo_goal_rotation=np.eye(3);self.angle_retry=0
        self.recovery_seeded_supervisor=None;self.approach_measured_history=[]
        self.recovery_command=None;self.recovery_command_stamp=None;self.servo_stall_history=[]

    def _execute(self,task,get_observation,move_robot,send_feedback,**kwargs):
        # SFP uses the already tested dwell repair. SC recovery is a separate
        # failure-specific diagnostic, not an untested change to both families.
        self.mode=self.requested_recovery if self.local.task_vector(task)[1]==1 else 'none'
        self.angle_retry=0;self.servo_goal_rotation=np.eye(3);self.servo_stall_history=[]
        return super()._execute(task,get_observation,move_robot,send_feedback,**kwargs)

    def _lock(self,task_vector,get_observation,move_robot,*,fresh_context):
        history=[]
        def observed():
            value=get_observation()
            if value is not None:
                stamp=self.local._control_sim_time(value)
                if not history or stamp>history[-1][0]:history.append((stamp,self.local.assembler.assemble_ros(value).copy()))
            return value
        accepted=super()._lock(task_vector,observed,move_robot,fresh_context=fresh_context)
        if accepted:self.approach_measured_history=history
        return accepted

    def _supervise(self,state,absolute):
        if self.recovery_seeded_supervisor is not self.supervisor:
            for stamp,measured in self.approach_measured_history:
                self.supervisor._append_history(measured_pose(measured),self._force(measured),'approach',self.route)
            self.recovery_seeded_supervisor=self.supervisor
            self._trace(event='backoff_approach_history_seeded',observed_samples=len(self.approach_measured_history),geometry_input=False)
        return super()._supervise(state,absolute)

    def _supervise_servo_stall(self,state,issued,point,stamp,force,depth):
        self.servo_stall_history.append((stamp,float(point[2,3])))
        self.servo_stall_history=[x for x in self.servo_stall_history if stamp-x[0]<=1.6]
        if len(self.servo_stall_history)<2:return
        elapsed=stamp-self.servo_stall_history[0][0]
        gain=point[2,3]-self.servo_stall_history[0][1]
        demand=float(np.linalg.norm(issued[:3,3]-state[:3]))
        if elapsed>=1.5 and gain<=.0003 and demand>=.003 and force>=3. and point[2,3]<depth-.0005:
            phase='contact' if point[2,3]>-.005 else 'alignment'
            self.supervisor._trigger(measured_pose(state),force,phase,self.route)
            decision=self.supervisor.step(pose=measured_pose(state),force_n=force,
                prior_command_m=self.last_command_m,tracking_error_m=demand,phase=phase,route=self.route)
            self._trace(event='loaded_servo_stall_recovery',sim_time=stamp,load_change_n=force,
                progress_window_s=elapsed,axial_gain_mm=float(gain*1000),tracking_error_mm=demand*1000,
                reason='sustained_actual_load_and_tracking_stall_before_estimated_seating',
                synthetic_force_used=False,geometry_input=False,level=decision.level)
            self.local.clear_pending_actions();raise RecoveryTransfer(decision)

    def _command(self,move_robot,matrix,*,owner):
        super()._command(move_robot,matrix,owner=owner)
        if owner in ('measured_path_backtrack','supervisor_lateral_exploration'):
            self.recovery_command=np.asarray(matrix).copy();self.recovery_command_stamp=self.local.last_observation_stamp

    def _lateral_retry(self,get_observation,move_robot,*,route_level):
        self.recovery_command=None
        if not super()._lateral_retry(get_observation,move_robot,route_level=route_level):return False
        if self.recovery_command is None:raise ValueError('Missing actual recovery command')
        self.local.clear_pending_actions();self.local.governor=None
        self.local.governor_seed=(self.recovery_command.copy(),self.recovery_command_stamp)
        self.servo_stall_history=[];self.angle_retry+=1
        # Six symmetric candidates cover every axis/sign; predeclared without
        # reading the current episode's true grasp, target pose or scoring.
        proposals=[(2.5,0,0),(-2.5,0,0),(0,2.5,0),(0,-2.5,0),(0,0,2.5),(0,0,-2.5)]
        if self.angle_retry>len(proposals):
            self._trace(event='recovery_abort',reason='bounded_angular_retry_budget_exhausted');return False
        self.servo_goal_rotation=Rotation.from_rotvec(np.deg2rad(proposals[self.angle_retry-1])).as_matrix()
        self._trace(event='backoff_reference_reseeded',sim_time=self.recovery_command_stamp,
            issued_tcp_target=self.recovery_command.tolist(),goal_rotation_offset_rotvec_deg=proposals[self.angle_retry-1],
            exploration_source='fixed_symmetric_angular_candidates_and_seeded_supervisor_lateral_retry_not_RL',geometry_input=False)
        return True

    def _visual_servo_insert(self,task,get_observation,move_robot,send_feedback,**kwargs):
        vector=self.local.task_vector(task);port=self.local.visual_port.copy();inverse=np.linalg.inv(port)
        grasp=np.eye(4);grasp[:3,3]=self.local.visual_body_tip;grasp[:3,:3]=self.local.visual_body_connector_rotation
        inverse_grasp=np.linalg.inv(grasp);depth=.0465 if vector[0]==1 else .0156
        seed,seed_stamp=self.local.governor_seed
        started=time.monotonic();first=None;last=None;stable=None;best=-float('inf');last_progress=None;reference=None;integral=None;governor=None;reason='visual_servo_wall_budget'
        while time.monotonic()-started<60 and time.monotonic()<self.episode_deadline-2:
            observation=get_observation()
            if observation is None:time.sleep(.002);continue
            stamp=self.local._control_sim_time(observation)
            if last is not None and stamp-last<.049:time.sleep(.002);continue
            control_started=time.perf_counter()
            dt=.05 if last is None else min(.1,stamp-last);last=stamp
            state=self.local.assembler.assemble_ros(observation);measured=matrix_from_xyz_xyzw(state[:7]);point=inverse@measured@grasp
            angle=Rotation.from_matrix(self.servo_goal_rotation.T@point[:3,:3]).magnitude();force=self._force(state)
            if first is None:
                first=stamp;last_progress=stamp;reference=np.array([0.,0.,point[2,3]])
                integral=(inverse@seed@grasp)[:3,3]-reference
                governor=InsertionReferenceGovernor(port,grasp,seed,measured,seed_stamp,
                    ReferenceLimits(forward_lead_m=.020,alignment_lateral_m=1.,alignment_angle_rad=np.pi,contact_load_n=float('inf')))
                self._trace(event='visual_servo_started',sim_time=stamp,geometry_input=False,actor_owned=False,
                    learned_actor_inference=False,required_seating_dwell_s=2.0,source='frozen RGB port/grasp and public robot state only')
            if point[2,3]>best+.0005:best=float(point[2,3]);last_progress=stamp
            if np.linalg.norm(state[26:29])>40 or (force>=8 and self.mode=='none'):
                reason='visual_servo_force_guard';break
            if stamp-first>35:reason='visual_servo_simulation_budget';break
            if stamp-last_progress>8:reason='visual_servo_no_measured_progress';break
            lateral=float(np.linalg.norm(point[:2,3]));aligned=bool(lateral<=.0005 and angle<=np.deg2rad(.5))
            if aligned:reference[2]=min(depth,reference[2]+.006*dt)
            integral=np.clip(integral+.6*dt*(reference-point[:3,3]),-.015,.015)
            desired_point=np.eye(4);desired_point[:3,3]=reference+integral;desired_point[:3,:3]=self.servo_goal_rotation
            desired=port@desired_point@inverse_grasp
            issued,diagnostic=governor.step(desired,measured,stamp,force)
            if self.mode!='none':
                self._supervise(state,issued)
                self._supervise_servo_stall(state,issued,point,stamp,force,depth)
            self._command(move_robot,issued,owner='image_only_insertion_servo_diagnostic')
            self._trace(event='visual_servo_step',sim_time=stamp,estimated_tip_port_m=point[:3,3].tolist(),
                estimated_orientation_error_deg=float(np.degrees(angle)),reference_tip_port_m=reference.tolist(),
                integral_port_m=integral.tolist(),load_change_n=force,aligned=aligned,
                goal_orientation_offset_deg=np.rad2deg(Rotation.from_matrix(self.servo_goal_rotation).as_rotvec()).tolist(),
                complete_control_step_ms=(time.perf_counter()-control_started)*1000,
                geometry_input=False,actor_owned=False,learned_actor_inference=False,governor=diagnostic)
            seated=(point[2,3]>=depth-.0002 and lateral<=.0005 and angle<=np.deg2rad(.75) and np.linalg.norm(state[7:10])<=.002)
            stable=(stamp if stable is None else stable) if seated else None
            if stable is not None and stamp-stable>=2.0:
                reason='visual_servo_public_estimated_seating';self._trace(event='visual_servo_finished',reason=reason,sim_time=stamp,geometry_input=False,physical_insertion_asserted=False);return True
        self._trace(event='visual_servo_finished',reason=reason,sim_time=last,geometry_input=False,physical_insertion_asserted=False)
        return False
