"""Two task-specific rule routes, one learned actor, measured-path recovery.

This candidate must pass autonomous development tests before promotion. No
scene file, object TF, scoring event or simulator contact enters decisions.
Supervisor moves are explicitly not actor actions or RL log-probabilities.
"""
import os
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from aic_model.policy import Policy,build_pose_from_vectors
from aic_control_interfaces.msg import TargetMode
from .RunLockedPortLocal import RunLockedPortLocal
from .RunPredictedPortLocal import RunPredictedPortLocal
from .hierarchical_recovery import HierarchicalRecovery,RecoveryConfig,MeasuredPose
from causal_force_reference import InitialLoadReference
from correlated_lateral_retry import lateral_retry_direction
from shared_port_tcp_geometry import matrix_from_xyz_xyzw
from sc_native_grasp_observation import SCNativeGraspObservation
from sfp_native_grasp_observation import SFPNativeGraspObservation


class RecoveryTransfer(Exception):
    def __init__(self,decision):
        super().__init__(decision.reason or decision.mode)
        self.decision=decision


class RuntimeBudgetReached(Exception):
    pass


class ObservationContractError(Exception):
    pass


class SupervisedLockedActor(RunLockedPortLocal):
    def __init__(self,parent_node,checkpoint):
        super().__init__(parent_node,checkpoint)
        self.supervisor_callback=None

    def _bound_transport_target(self,absolute,task_vector,state):
        if self.supervisor_callback is not None:
            self.supervisor_callback(state,absolute)
        return absolute


def measured_pose(state):
    q=state[3:7]
    return MeasuredPose(tuple(state[:3]),(q[3],q[0],q[1],q[2]))


class RunRuleThenSharedRecovery(Policy):
    def _make_local_actor(self, checkpoint):
        return SupervisedLockedActor(self.parent, checkpoint)

    def __init__(self,parent_node):
        super().__init__(parent_node)
        self.parent=parent_node
        self.local=self._make_local_actor(os.environ['AIC_OFFICIAL_TEACHER_LOCAL_ACTOR_CHECKPOINT'])
        if set(self.local.trained_families)!={'sc','sfp'}:
            raise ValueError('This pipeline requires a shared actor trained on BOTH task families')
        self.mode=os.environ.get('AIC_SHARED_RECOVERY_MODE','hierarchical')
        if self.mode not in ('none','local','hierarchical'):raise ValueError('Unknown matched recovery arm')
        self.local.control_hz=20.;self.local.control_clock='simulation'
        self.local.execute_steps=4;self.local.execute_complete_chunk=True
        self.local.max_translation=.06;self.local.max_rotation=.15
        self.local.cartesian_stiffness=[90,90,90,50,50,50]
        self.local.cartesian_damping=[50,50,50,20,20,20]
        self.total_budget_s=float(os.environ.get('AIC_SHARED_RECOVERY_SIM_BUDGET_S','180'))
        self.rule=None
        self.episode_index=0
        self.perception_latencies_ms=[]
        self.exploration_seed=int(os.environ.get('AIC_SHARED_RECOVERY_EXPLORATION_SEED','20261016'))
        self._warm_actor()

    @torch.inference_mode()
    def _warm_actor(self):
        actor=self.local.actor;actor.eval();device=self.local.device
        state=actor.state_mean[None];pose=actor.pose_mean[None]
        history=torch.cat((state,pose,torch.zeros(1,1,device=device)),1)[:,None].expand(-1,4,-1)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            actor(torch.zeros(1,3,3,224,224,dtype=torch.uint8,device=device),state,
                torch.tensor([[1.,0.,1.,0.,1.,0.,0.,0.,0.,1.]],device=device),
                external_pose=pose,external_mask=torch.ones(1,dtype=torch.bool,device=device),
                physical_tip=actor.tip_mean[None],tip_mask=torch.ones(1,dtype=torch.bool,device=device),
                local_history=history if actor.use_local_history else None)
        if device.type=='cuda':torch.cuda.synchronize(device)

    @torch.inference_mode()
    def _warm_grasp(self):
        # Warm the image network only. No fake geometry/pose is bound and no
        # acquisition approval is inferred from these synthetic pixels.
        self.grasp.model.eval()
        self.grasp.model(torch.zeros(3,3,192,192,device=self.local.device),
                         camera_id=torch.eye(3,device=self.local.device))
        if self.local.device.type=='cuda':torch.cuda.synchronize(self.local.device)

    def _trace(self,**fields):
        self.local._trace(**fields)

    def _command(self,move_robot,matrix,*,owner):
        if hasattr(self.parent,'_target_mode'):
            self.parent._target_mode=TargetMode.MODE_CARTESIAN
        self.local.set_pose_target(move_robot=move_robot,
            pose=build_pose_from_vectors(matrix[:3,3],Rotation.from_matrix(matrix[:3,:3]).as_quat()),
            frame_id='base_link',stiffness=self.local.cartesian_stiffness,damping=self.local.cartesian_damping)
        self._trace(event='supervisor_command',owner=owner,actor_owned=False,
                    absolute_tcp_target=matrix.tolist(),geometry_input=False)

    def _force(self,state):
        return self.force.measure(state[26:29],state[3:7])['load_change_norm_n']

    def _initial_force_reference(self,get_observation,move_robot):
        self.force=InitialLoadReference(minimum_quiet_span_s=getattr(self,'initial_force_minimum_quiet_span_s',0.))
        last=-float('inf');last_force=-float('inf');started=time.monotonic();hold=None
        while time.monotonic()-started<20:
            observation=get_observation()
            if observation is None:time.sleep(.005);continue
            sim=self.local._control_sim_time(observation)
            force_stamp=observation.wrist_wrench.header.stamp
            force_time=force_stamp.sec+force_stamp.nanosec*1e-9
            if getattr(self,'unique_initial_force_samples',False):
                if force_time<=last_force or abs(force_time-sim)>.1:
                    time.sleep(.002);continue
            elif sim-last<.05:time.sleep(.002);continue
            state=self.local.assembler.assemble_ros(observation)
            if hold is None:hold=matrix_from_xyz_xyzw(state[:7])
            self._command(move_robot,hold,owner='initial_free_hold')
            last=sim
            if force_time<=last_force:continue
            last_force=force_time
            if self.force.add_initial_free_hold(stamp_s=force_time,force_sensor_n=state[26:29],
                    tcp_quaternion_xyzw=state[3:7],linear_speed_mps=np.linalg.norm(state[7:10]),
                    angular_speed_radps=np.linalg.norm(state[10:13])):
                self._trace(event='initial_force_reference_frozen',sim_time=sim,
                    base_force_n=self.force.reference.tolist(),initial_free_hold_assumed=True,
                    quiet_sample_span_s=self.force.timestamps[-1]-self.force.timestamps[0],
                    quiet_samples_observed=len(self.force.observations),median_force_samples=self.force.samples,
                    oracle_contact_force=False)
                return True
        return False

    def _fresh_report(self,observation,task_vector):
        state=self.local.assembler.assemble_ros(observation)
        sim=self.local._control_sim_time(observation)
        images={c:self.rule.native(getattr(observation,c+'_image')) for c in ('center','left','right')}
        port=int(np.argmax(task_vector[2:4]))
        if task_vector[1]==1:
            report=self.rule.verifier.inspect(images,state,port)
        else:
            card=int(np.argmax(task_vector[4:9]))
            result=self.rule.observer.predict_batch([({c:im[...,::-1].copy() for c,im in images.items()},state,card,port)])[0]
            report=dict(valid=False,runtime_true_geometry=False,reason='fresh_requested_pose_rejected')
            if result['accepted']:
                point=np.asarray(result['predicted_base_m']);prior=self.rule.priors['tasks'][f'card{card}_port{port}']
                within=bool(np.all(point>=prior['potential_min_base_m']) and np.all(point<=prior['potential_max_base_m']))
                report.update(valid=within,reason='fresh_requested_native_pose' if within else 'outside_training_potential_region',
                    predicted_opening_base_m=point.tolist(),predicted_port_quat_xyzw=result['predicted_orientation_base_xyzw'])
        return report,images,state,sim

    def _lock(self,task_vector,get_observation,move_robot,*,fresh_context):
        self.local.clear_pending_actions();self.local.visual_port=None;self.local.visual_body_tip=None
        self.local.robot_history=[]
        if fresh_context:
            if task_vector[1]==1:self.rule.verifier.reset_context()
            else:self.rule.observer.reset_context()
        previous=None;predictions=[];last=-float('inf');started=time.monotonic()
        while time.monotonic()-started<20:
            observation=get_observation()
            if observation is None:time.sleep(.005);continue
            sim=self.local._control_sim_time(observation)
            if sim-last<.15:time.sleep(.002);continue
            last=sim;inference_started=time.perf_counter()
            report,images,state,sim=self._fresh_report(observation,task_vector)
            self._command(move_robot,matrix_from_xyz_xyzw(self.acquisition_state[:7]),owner='fresh_visual_acquisition_hold')
            if not report['valid']:
                self.perception_latencies_ms.append((time.perf_counter()-inference_started)*1000)
                previous=None;predictions=[];continue
            port=matrix_from_xyz_xyzw([*report['predicted_opening_base_m'],*report['predicted_port_quat_xyzw']])
            tcp=matrix_from_xyz_xyzw(state[:7]);relative=np.linalg.inv(port)@tcp
            gap=-relative[2,3]*1000
            if task_vector[0]==1 and gap<120:previous=None;predictions=[];continue
            if previous is not None:
                spread=np.linalg.norm(port[:3,3]-previous[:3,3])
                angle=(Rotation.from_matrix(previous[:3,:3]).inv()*Rotation.from_matrix(port[:3,:3])).magnitude()
                if spread>.001 or angle>np.deg2rad(1):predictions=[]
            lock_pose=port.copy()
            if previous is not None:
                lock_pose[:3,3]=(port[:3,3]+previous[:3,3])/2
                lock_pose[:3,:3]=Rotation.from_matrix(np.stack([port[:3,:3],previous[:3,:3]])).mean().as_matrix()
            previous=port
            kwargs=dict(estimated_TCP_axial_gap_mm=float(gap)) if task_vector[0]==1 else {}
            prediction=self.grasp.predict(images,**kwargs)
            duration=(time.perf_counter()-inference_started)*1000
            self.perception_latencies_ms.append(duration)
            self._trace(event='shared_perception_latency',sim_time=sim,total_ms=duration,
                includes='three native RGB conversion + requested pose + body-tip estimate + hold publication')
            predictions=(predictions+[np.asarray(prediction['predicted_tip_position_tcp_m'])])[-3:]
            if len(predictions)<3:continue
            if getattr(self,'joint_grasp_acquisition',False) and not report.get('connector_rotation_consensus_accepted',False):continue
            required_far=getattr(self,'required_SC_far_context_frames',0) if task_vector[1]==1 else 0
            if required_far and len(getattr(self,'far_grasp_frames',[]))<required_far:
                self._trace(event='joint_acquisition_waiting_for_quiet_far_context',sim_time=sim,
                    available_frames=len(getattr(self,'far_grasp_frames',[])),required_frames=required_far,
                    geometry_input=False)
                continue
            body=np.median(predictions,axis=0)
            spread=max(np.linalg.norm(x-body) for x in predictions)
            if spread>.0005:continue
            # Deployment handoff owns the whole150mm approach. Do not silently
            # start this actor outside its declared180mm buffered envelope.
            locked_relative=np.linalg.inv(lock_pose)@tcp
            tip=locked_relative[:3,3]+locked_relative[:3,:3]@body
            # Legacy learned actors retain their exact 100–185 mm envelope.
            # An explicit controller can declare a shorter SFP approach; the
            # separate RGB grasp-estimator TCP-gap gate above still applies.
            minimum_tip_gap=(getattr(self,'SFP_minimum_shared_tip_standoff_m',.100)
                             if task_vector[0]==1 else .100)
            if not .050<=minimum_tip_gap<=.100:
                raise ValueError('Invalid explicit SFP acquisition envelope')
            maximum_tip_lateral=(getattr(self,'SFP_maximum_shared_tip_lateral_m',.030)
                                 if task_vector[0]==1 else .030)
            if not .030<=maximum_tip_lateral<=.060:
                raise ValueError('Invalid explicit SFP lateral acquisition envelope')
            if not(-.185<=tip[2]<=-minimum_tip_gap and np.linalg.norm(tip[:2])<=maximum_tip_lateral):
                self._trace(event='shared_lock_rejected',reason='outside_approach_envelope',tip_port_m=tip.tolist())
                return False
            report=dict(report,predicted_opening_base_m=lock_pose[:3,3].tolist(),
                predicted_port_quat_xyzw=Rotation.from_matrix(lock_pose[:3,:3]).as_quat().tolist(),
                lock_measurements=2,lock_rule='same two-consistent-pose mean as BC frame preparation')
            self.local.bind_visual_handoff(report,sim);self.local.bind_predicted_body_tip(body)
            self.acquisition_state=state.copy()
            self._trace(event='shared_visual_lock',sim_time=sim,report=report,
                body_tip_position_tcp_m=body.tolist(),grasp_spread_mm=float(spread*1000),
                requested_task=np.asarray(task_vector).tolist(),fresh_context=fresh_context,
                geometry_input=False,actor_owns_full_150mm=True)
            return True
        return False

    def _supervise(self,state,absolute):
        force=self._force(state)
        relative=np.linalg.inv(self.local.visual_port)@matrix_from_xyz_xyzw(state[:7])
        tip=relative[:3,3]+relative[:3,:3]@self.local.visual_body_tip
        phase='contact' if tip[2]>-.005 else ('alignment' if tip[2]>-.04 else 'approach')
        demand=0. if self.last_target is None else float(np.linalg.norm(self.last_target-state[:3]))
        decision=self.supervisor.step(pose=measured_pose(state),force_n=force,
            prior_command_m=self.last_command_m,tracking_error_m=demand,phase=phase,route=self.route)
        self._trace(event='recovery_observation',mode=decision.mode,level=decision.level,
            measured_state=state.tolist(),load_change_n=force,prior_command_m=self.last_command_m,
            tracking_error_m=demand,phase=phase,route=self.route,actor_owned=decision.actor_owned)
        if not decision.actor_owned:
            self.local.clear_pending_actions()
            raise RecoveryTransfer(decision)
        self.last_target=absolute[:3,3].copy()
        self.last_command_m=float(np.linalg.norm(absolute[:3,3]-state[:3]))

    def _retreat(self,transfer,get_observation,move_robot):
        if transfer.decision.mode=='abort':return False
        route_level=transfer.decision.level=='route'
        self.blocked_pose=self.supervisor.previous_pose
        if route_level:
            # Extend the selected measured reverse path to the acquisition
            # checkpoint. Returning only60mm is insufficient to legally
            # reacquire the SFP plug offset before the next150mm approach.
            if not self.supervisor.extend_route_to_acquisition():return False
            self.local.visual_port=None;self.local.visual_body_tip=None
        else:
            # A12mm retreat from an already partly inserted SFP plug can
            # still leave it inside the opening. Extend along measured history
            # until its OBSERVATION-estimated tip is at least5mm outside.
            inverse=np.linalg.inv(self.local.visual_port);body=self.local.visual_body_tip
            def outside(p):
                q=p.quat_wxyz;m=matrix_from_xyz_xyzw([*p.xyz,q[1],q[2],q[3],q[0]])
                # Include arrival tolerance and estimator margin, so reaching
                # the selected checkpoint really permits the5mm-clear retry.
                return (inverse@np.r_[m[:3,3]+m[:3,:3]@body,1.])[2]<=-.008
            if not outside(self.supervisor.anchor):
                candidates=[i for i in range(self.supervisor.anchor_history_index,-1,-1)
                    if self.supervisor.history[i][1]<=self.supervisor.config.clear_force_n
                    and outside(self.supervisor.history[i][0])]
                if not candidates:
                    self._trace(event='recovery_abort',reason='no_measured_outside_opening_checkpoint');return False
                i=candidates[0]
                self.supervisor.path=[x[0] for x in reversed(self.supervisor.history[i:])]
                self.supervisor.path_index=0;self.supervisor.anchor=self.supervisor.history[i][0]
                self.supervisor.anchor_history_index=i
        started=time.monotonic();last=-float('inf')
        while time.monotonic()-started<60:
            observation=get_observation()
            if observation is None:time.sleep(.005);continue
            sim=self.local._control_sim_time(observation)
            if sim-self.insertion_start_sim>self.total_budget_s:return False
            if sim-last<.05:time.sleep(.002);continue
            last=sim;state=self.local.assembler.assemble_ros(observation)
            decision=self.supervisor.step(pose=measured_pose(state),force_n=self._force(state),
                prior_command_m=0.,tracking_error_m=0.,phase='recovery',route=self.route)
            if decision.mode=='abort':
                self._trace(event='recovery_abort',reason=decision.reason,sim_time=sim);return False
            if decision.mode=='retry':
                self._trace(event='measured_retreat_complete',sim_time=sim,level=decision.level,
                    total_backoffs=decision.total_backoffs,route_resets=decision.route_resets)
                return True
            if decision.target is not None:
                p=decision.target;q=p.quat_wxyz
                target=matrix_from_xyz_xyzw([*p.xyz,q[1],q[2],q[3],q[0]])
                self._command(move_robot,target,owner='measured_path_backtrack')
        self._trace(event='recovery_abort',reason='retreat_timeout');return False

    def _lateral_retry(self,get_observation,move_robot,*,route_level):
        observation=get_observation()
        if observation is None:return False
        state=self.local.assembler.assemble_ros(observation);start=matrix_from_xyz_xyzw(state[:7])
        inverse=np.linalg.inv(self.local.visual_port);body=self.local.visual_body_tip
        start_tip=(inverse@np.r_[start[:3,3]+start[:3,:3]@body,1.])[:3]
        if start_tip[2]>-.005:
            self._trace(event='recovery_abort',reason='lateral_retry_while_plug_may_be_inside');return False
        retreat=start[:3,3]-np.asarray(self.blocked_pose.xyz)
        direction=lateral_retry_direction(retreat,self.local.visual_port[:3,2],self.exploration_rng)
        amplitude=.010 if route_level else .003
        # Keep the proposed retry within the buffered approach envelope. Scale
        # the sampled displacement; preserve its orthogonality to the retreat.
        while amplitude>=.0005:
            tip=start_tip+inverse[:3,:3]@(direction*amplitude)
            if np.linalg.norm(tip[:2])<=.025 and tip[2]<=-.005:break
            amplitude*=.5
        if amplitude<.0005 or (route_level and amplitude<.003):return False
        goal=start.copy();goal[:3,3]+=direction*amplitude
        started=time.monotonic();last=-float('inf');arrival=0
        self.local.clear_pending_actions()
        while time.monotonic()-started<15:
            observation=get_observation()
            if observation is None:time.sleep(.005);continue
            sim=self.local._control_sim_time(observation)
            if sim-self.insertion_start_sim>self.total_budget_s:return False
            if sim-last<.05:time.sleep(.002);continue
            last=sim;state=self.local.assembler.assemble_ros(observation);current=matrix_from_xyz_xyzw(state[:7])
            force=self._force(state)
            if force>=self.supervisor.config.trigger_force_n:
                self._trace(event='recovery_abort',reason='new_load_during_lateral_retry',load_change_n=force);return False
            self.supervisor._append_history(measured_pose(state),force,'supervisor_lateral_retry',self.route)
            displacement=goal[:3,3]-state[:3];distance=np.linalg.norm(displacement)
            arrival=arrival+1 if distance<.00075 and np.linalg.norm(state[7:10])<.005 else 0
            if arrival>=3:
                moved=float(np.linalg.norm(state[:3]-start[:3,3]))
                if route_level and moved<.003:return False
                self._trace(event='lateral_retry_complete',sim_time=sim,route_level=route_level,
                    measured_displacement_m=(state[:3]-start[:3,3]).tolist(),sampled_direction_base=direction.tolist(),
                    amplitude_m=amplitude,seed=self.exploration_seed,actor_owned=False,
                    exploration_source='supervisor_coherent_direction_not_RL',geometry_input=False)
                self.last_measured_lateral_retry_m=moved
                return True
            target=current.copy();target[:3,3]+=displacement*min(1.,.001/max(distance,1e-9))
            target[:3,:3]=start[:3,:3]
            self._command(move_robot,target,owner='supervisor_lateral_exploration')
        return False

    def insert_cable(self,task,get_observation,move_robot,send_feedback,**kwargs):
        self.episode_index+=1
        self.local.reset_for_task(task);self.local.clear_pending_actions()
        self.local.visual_port=None;self.local.visual_body_tip=None
        self.local.local_terminal_reason=None;self.local.supervisor_callback=None
        self.rule=None
        # The official task budget includes transport and acquisition. Never
        # continue commanding after assets are removed at the task deadline.
        self.episode_deadline=time.monotonic()+max(1.,float(task.time_limit)-5.)
        self.perception_latencies_ms=[]
        self.pipeline_phase='initializing';self.pipeline_outcome=False
        self.retained_terminal=None;self.retained_native=None;self.retained_wall=None
        def checked_observation():
            observation=get_observation()
            if observation is not None:
                sim=self.local._control_sim_time(observation)
                if self.retained_terminal is None or sim>self.retained_terminal['sim_time']:
                    self.retained_terminal=dict(sim_time=sim,
                        observation_wall_time_ns=time.time_ns(),
                        state=self.local.assembler.assemble_ros(observation).tolist())
                    self.retained_native=observation;self.retained_wall=time.monotonic()
                header=observation.wrist_wrench.header
                force_time=header.stamp.sec+header.stamp.nanosec*1e-9
                if header.frame_id.lstrip('/')!='ati/tool_link' or abs(sim-force_time)>.1:
                    self.local.clear_pending_actions()
                    self._trace(event='terminal_before_return',reason='invalid_or_stale_wrist_force',
                        sim_time=sim,state=self.local.assembler.assemble_ros(observation).tolist(),
                        wrench_frame=header.frame_id,wrench_age_s=sim-force_time)
                    raise ObservationContractError()
            if time.monotonic()>=self.episode_deadline:
                self.local.clear_pending_actions()
                self._trace(event='terminal_before_return',reason='whole_task_budget',
                    sim_time=self.local._control_sim_time(observation) if observation is not None else None,
                    state=self.local.assembler.assemble_ros(observation).tolist() if observation is not None else None,
                    post_reset_geometry_used=False)
                raise RuntimeBudgetReached()
            return observation
        try:
            self.pipeline_outcome=self._execute(task,checked_observation,move_robot,send_feedback,**kwargs)
            return self.pipeline_outcome
        except (RuntimeBudgetReached,ObservationContractError):
            return False
        except Exception as error:
            self._trace(event='pipeline_exception',stage=self.pipeline_phase,
                error_type=type(error).__name__,message=str(error),
                implementation_failure=True,geometry_input=False)
            raise
        finally:
            self.local.clear_pending_actions()
            # Save the already received pre-return observation. Never call a
            # sensor service after action return/reset to invent terminal data.
            terminal_images={};image_stamps={}
            if self.retained_native is not None and self.local.trace_path:
                folder=Path(self.local.trace_path).parent/'terminal_native_rgb'/f'action_{self.episode_index:05d}'
                folder.mkdir(parents=True,exist_ok=False)
                for camera in ('center','left','right'):
                    message=getattr(self.retained_native,camera+'_image')
                    image=RunPredictedPortLocal.raw_rgb(message);path=folder/(camera+'.jpg')
                    if cv2.imwrite(str(path),image[...,::-1],[cv2.IMWRITE_JPEG_QUALITY,86]):
                        terminal_images[camera]=str(path)
                    image_stamps[camera]=message.header.stamp.sec*1_000_000_000+message.header.stamp.nanosec
            self._trace(event='terminal_before_return',reason='shared_pipeline_'+self.pipeline_phase,
                **(self.retained_terminal or dict(sim_time=None,state=None)),
                observation_source='retained_public_observation_before_return',
                observation_age_wall_s=(time.monotonic()-self.retained_wall if self.retained_wall is not None else None),
                terminal_images=terminal_images,image_stamps_ns=image_stamps,
                attempt_acknowledged=bool(self.pipeline_outcome),physical_insertion_asserted=False,
                underlying_actor_terminal_reason=self.local.local_terminal_reason,
                visual_port_matrix=(self.local.visual_port.tolist() if self.local.visual_port is not None else None),
                body_tip_position_tcp_m=(self.local.visual_body_tip.tolist() if self.local.visual_body_tip is not None else None),
                post_reset_geometry_used=False,task_id=str(task.id),episode_index=self.episode_index)
            if self.perception_latencies_ms:
                self._trace(event='shared_perception_latency_summary',calls=len(self.perception_latencies_ms),
                    p50_p95_p99_ms=np.percentile(self.perception_latencies_ms,[50,95,99]).tolist())

    def _execute(self,task,get_observation,move_robot,send_feedback,**kwargs):
        self.exploration_rng=np.random.default_rng(self.exploration_seed)
        task_vector=self.local.task_vector(task)
        if task_vector[1]==1:
            from .RunNative5RequestedRuleTransport import RunNative5RequestedRuleTransport
            self.rule=RunNative5RequestedRuleTransport(self.parent)
            self.grasp=SCNativeGraspObservation(os.environ['AIC_LOCKED_GRASP_CHECKPOINT'],str(self.local.device))
        else:
            from .RunSfpRequestedRuleTransport import RunSfpRequestedRuleTransport
            self.rule=RunSfpRequestedRuleTransport(self.parent)
            self.grasp=SFPNativeGraspObservation(os.environ['AIC_SFP_GRASP_CHECKPOINT'],os.environ['AIC_SFP_GRASP_APPROVAL'],str(self.local.device))
            # SC helper supplies the native ROS image conversion for both.
            from .RunRuleTransport import RunRuleTransport
            self.rule.native=RunRuleTransport.native
        self._warm_grasp()
        self.pipeline_phase='initial_force_reference'
        if not self._initial_force_reference(get_observation,move_robot):return False
        self.pipeline_phase='rule_transport'
        self.rule.insert_cable(task,get_observation,move_robot,send_feedback,**kwargs)
        if not self.rule.rule_final_valid or self.rule.rule_failure:return False
        observation=get_observation()
        if observation is None:return False
        self.acquisition_state=self.local.assembler.assemble_ros(observation)
        self.pipeline_phase='fresh_visual_acquisition'
        if not self._lock(task_vector,get_observation,move_robot,fresh_context=True):return False
        self.supervisor=HierarchicalRecovery(RecoveryConfig(max_total_backoffs=8,max_history=4096,
            local_attempts_before_route=2 if self.mode=='hierarchical' else 100))
        self.last_target=None;self.last_command_m=0.;self.route='acquisition_0'
        self.supervisor.step(pose=measured_pose(self.acquisition_state),
            force_n=self._force(self.acquisition_state),prior_command_m=0.,
            phase='acquisition_checkpoint',route=self.route)
        self.insertion_start_sim=self.local._control_sim_time(get_observation())
        self.local.supervisor_callback=self._supervise if self.mode!='none' else None
        while True:
            current=get_observation()
            if current is None:return False
            remaining=self.total_budget_s-(self.local._control_sim_time(current)-self.insertion_start_sim)
            if remaining<=0:return False
            self.local.max_simulation_sec=remaining
            remaining_wall=self.episode_deadline-time.monotonic()
            if remaining_wall<=2:return False
            # Let the normal bounded actor attempt return and retain its final
            # observation before the outer deadline. Otherwise the equal
            # deadlines always turn a completed attempt into an API failure.
            # Physical insertion is still determined by independent scoring.
            self.local.max_runtime_sec=remaining_wall-1.
            self.pipeline_phase='actor'
            try:return self.local.insert_cable(task,get_observation,move_robot,send_feedback,**kwargs)
            except RecoveryTransfer as transfer:
                self.last_target=None;self.last_command_m=0.
                self.pipeline_phase='measured_backtrack'
                if not self._retreat(transfer,get_observation,move_robot):return False
                if transfer.decision.level=='route':
                    self.pipeline_phase='fresh_reacquisition'
                    if not self._lock(task_vector,get_observation,move_robot,fresh_context=True):return False
                self.pipeline_phase='supervisor_lateral_retry'
                if not self._lateral_retry(get_observation,move_robot,route_level=transfer.decision.level=='route'):
                    return False
                if transfer.decision.level=='route':
                    # Change route identity only AFTER actual measured lateral
                    # motion. Renaming the same failed corridor is not a reset.
                    self.route=self.supervisor.confirm_new_corridor(
                        f'acquisition_{self.supervisor.route_resets}',self.last_measured_lateral_retry_m)
                # The same actor resumes. Clear pending chunks and all robot
                # history at every control transfer; never retare contact.
                self.local.clear_pending_actions();self.local.robot_history=[]
