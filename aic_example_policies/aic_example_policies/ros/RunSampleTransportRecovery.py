"""Observation-only short transport retreat, lateral retry, and route resume.

The large measured route reset is retained as an unused historical diagnostic;
ordinary transport contact recovery is bounded to a short local maneuver.
This is rule recovery, not RL or an expert action label.
"""
import copy
import time
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from aic_model.policy import build_pose_from_vectors
from aic_control_interfaces.msg import TargetMode

from .RunSampleConservativeTransport import RunSampleConservativeTransport
from shared_port_tcp_geometry import matrix_from_xyz_xyzw

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from causal_force_reference import InitialLoadReference
from transport_local_recovery import (TransportLoadStallMonitor,
    measured_approach_axis,horizontal_lateral_axis)


class TransportRecoveryBudget(Exception):
    pass


class TransportLocalContact(Exception):
    pass


class RunSampleTransportRecovery(RunSampleConservativeTransport):
    def __init__(self,parent_node):
        self._record_route=False;self._route_history=[];self._route_forces=[];self._route_clear_checkpoint=None;self._route_force_stop=False
        self._transport_attempt=0
        self._resume_transport=False;self._resume_pose=None;self._resume_sim_time=None;self._force_reference=None
        self._contact_monitor=TransportLoadStallMonitor()
        super().__init__(parent_node)

    def reset_for_task(self,task):
        if self._resume_transport:
            # The parent starts each bounded control call with reset_for_task.
            # A local retry is still the SAME episode and SAME route stage.
            self.clear_pending_actions();self.assembler.reset()
            self.rule_failure=None;self.rule_final_valid=False
            self.reference=self._resume_pose.copy();self.compensation[:]=0
            self.best_error=float('inf');self.last_progress=self._resume_sim_time
            self.stage_streak=0;self.hold=None
            self._contact_monitor.reset();self._resume_transport=False
            return
        super().reset_for_task(task)

    def predict_chunk(self,observation,task_vector,*,decision_time_s=None):
        targets,estimated,state=super().predict_chunk(observation,task_vector,
            decision_time_s=decision_time_s)
        if (self._record_route and self._force_reference is not None
                and self.rule_failure is None and not self.rule_final_valid):
            sim=self._control_sim_time(observation)
            load=self._force_reference.measure(state[26:29],state[3:7])['load_change_norm_n']
            name,xyz,_=self.stages[self.index]
            error=float(np.linalg.norm(np.asarray(xyz)-state[:3]))
            if self._contact_monitor.update(sim_time=sim,stage=name,
                    goal_error_m=error,load_change_n=load):
                self._trace(event='transport_local_contact_detected',sim_time=sim,
                    stage=name,load_change_n=load,goal_error_mm=error*1000,
                    reason='sustained_load_and_lost_measured_waypoint_progress',
                    oracle_contact_used=False,geometry_input=False)
                self.clear_pending_actions()
                raise TransportLocalContact()
        return targets,estimated,state

    def _calibrate_transport_force(self,get_observation,move_robot):
        self.assembler.reset()
        reference=InitialLoadReference(samples=10,minimum_quiet_span_s=.35)
        last=-float('inf');started=time.monotonic();hold=None
        while time.monotonic()-started<8:
            observation=get_observation()
            if observation is None:time.sleep(.005);continue
            sim=self._control_sim_time(observation)
            if sim-last<.05:time.sleep(.002);continue
            last=sim;state=self.assembler.assemble_ros(observation)
            if hold is None:hold=matrix_from_xyz_xyzw(state[:7])
            self._command_local(move_robot,hold,owner='transport_force_free_hold',state=state,sim=sim)
            stamp=observation.wrist_wrench.header.stamp
            force_time=stamp.sec+stamp.nanosec*1e-9
            if reference.last_stamp is not None and force_time<=reference.last_stamp:continue
            if abs(sim-force_time)>.1:continue
            if reference.add_initial_free_hold(stamp_s=force_time,
                    force_sensor_n=state[26:29],tcp_quaternion_xyzw=state[3:7],
                    linear_speed_mps=np.linalg.norm(state[7:10]),
                    angular_speed_radps=np.linalg.norm(state[10:13])):
                self._force_reference=reference
                self._trace(event='transport_initial_load_reference_frozen',sim_time=sim,
                    initial_reference_base_n=reference.reference.tolist(),
                    quiet_samples=reference.samples,oracle_contact_used=False,geometry_input=False)
                return True
        self._trace(event='transport_local_recovery_unavailable',reason='no_quiet_initial_force_reference')
        return False

    def _command_local(self,move_robot,goal,*,owner,state,sim):
        if hasattr(self._parent_node,'_target_mode'):
            self._parent_node._target_mode=TargetMode.MODE_CARTESIAN
        stiffness,damping=self._cartesian_gains()
        self.set_pose_target(move_robot=move_robot,
            pose=build_pose_from_vectors(goal[:3,3],Rotation.from_matrix(goal[:3,:3]).as_quat()),
            frame_id='base_link',stiffness=stiffness,damping=damping)
        self._trace(event='transport_local_recovery_command',sim_time=sim,owner=owner,
            state=state.tolist(),absolute_tcp_target=goal.tolist(),actor_owned=False,
            geometry_input=False)

    def _move_local(self,get_observation,move_robot,goal,*,owner,max_duration_s,
                    maximum_load_change_n,clear_load_change_n=None,
                    movement_axis=None,orthogonal_axis=None,minimum_projected_m=None):
        observation=get_observation()
        if observation is None:return None
        start_sim=self._control_sim_time(observation)
        state=self.assembler.assemble_ros(observation)
        start=matrix_from_xyz_xyzw(state[:7]);last=-float('inf');started=time.monotonic()
        distance=float(np.linalg.norm(goal[:3,3]-start[:3,3]));stable=0;integral=np.zeros(3)
        while time.monotonic()-started<max_duration_s:
            observation=get_observation()
            if observation is None:time.sleep(.005);continue
            sim=self._control_sim_time(observation)
            if sim-last<.05:time.sleep(.002);continue
            dt=.05 if last<0 else min(.1,sim-last);last=sim
            state=self.assembler.assemble_ros(observation)
            measured=matrix_from_xyz_xyzw(state[:7])
            load=self._force_reference.measure(state[26:29],state[3:7])['load_change_norm_n']
            if load>maximum_load_change_n:
                self._trace(event='transport_local_recovery_abort',sim_time=sim,
                    reason='load_growth',owner=owner,load_change_n=load,geometry_input=False)
                return None
            remaining=float(np.linalg.norm(goal[:3,3]-measured[:3,3]))
            clear=clear_load_change_n is None or load<=clear_load_change_n
            achieved=remaining<=.0015
            if minimum_projected_m is not None:
                displacement=measured[:3,3]-start[:3,3]
                projected=float(np.dot(displacement,movement_axis))
                orthogonal=float(np.dot(displacement,orthogonal_axis))
                achieved=(projected>=minimum_projected_m and abs(orthogonal)<=.003)
            stable=stable+1 if achieved and clear and np.linalg.norm(state[7:10])<.008 else 0
            if stable>=2:return measured,state,sim
            fraction=min(1.,max(0.,sim-start_sim)*.025/max(distance,1e-9))
            nominal=start[:3,3]+fraction*(goal[:3,3]-start[:3,3])
            integral=np.clip(integral+.5*dt*(nominal-state[:3]),-.004,.004)
            command=start.copy();command[:3,3]=nominal+integral
            self._command_local(move_robot,command,owner=owner,state=state,sim=sim)
        self._trace(event='transport_local_recovery_abort',reason='measured_arrival_timeout',
            owner=owner,geometry_input=False)
        return None

    def _recover_local(self,get_observation,move_robot,*,attempt):
        observation=get_observation()
        if observation is None:return False
        state=self.assembler.assemble_ros(observation);sim=self._control_sim_time(observation)
        current=matrix_from_xyz_xyzw(state[:7])
        load=self._force_reference.measure(state[26:29],state[3:7])['load_change_norm_n']
        if load>45.:
            self._trace(event='transport_local_recovery_abort',sim_time=sim,
                reason='too_much_initial_load',load_change_n=load,geometry_input=False)
            return False
        positions=[p[:3,3] for p in self._route_history[-40:]]+[current[:3,3]]
        try:axial=measured_approach_axis(positions)
        except ValueError:
            self._trace(event='transport_local_recovery_abort',sim_time=sim,
                reason='missing_measured_approach_direction',geometry_input=False)
            return False
        lateral=horizontal_lateral_axis(axial,1 if attempt==0 else -1)
        self._trace(event='transport_local_recovery_started',sim_time=sim,
            attempt=attempt+1,load_change_n=load,approach_axis_base=axial.tolist(),
            lateral_axis_base=lateral.tolist(),retreat_limit_mm=12.,
            oracle_contact_used=False,geometry_input=False)
        retreat=None
        for amount in (.008,.012):
            goal=current.copy();goal[:3,3]-=axial*amount
            retreat=self._move_local(get_observation,move_robot,goal,
                owner='transport_short_measured_retreat',max_duration_s=8.,
                maximum_load_change_n=max(30.,load+8.))
            # A failed move can mean renewed load or no measured arrival.
            # Never continue commanding after either failure.
            if retreat is None:return False
            measured_state=retreat[1]
            remaining_load=self._force_reference.measure(
                measured_state[26:29],measured_state[3:7])['load_change_norm_n']
            if remaining_load<=8.:break
        if retreat is None:return False
        if remaining_load>8.:
            # Cable load can relax after measured motion stops. Hold the
            # already reached retreat pose briefly; do not retreat farther.
            hold=retreat[0].copy();last=-float('inf');start_sim=retreat[2]
            settle_wall=time.monotonic()
            clear_streak=0
            while time.monotonic()-settle_wall<3.:
                observation=get_observation()
                if observation is None:time.sleep(.005);continue
                sim=self._control_sim_time(observation)
                if sim-last<.05:time.sleep(.002);continue
                last=sim;state=self.assembler.assemble_ros(observation)
                remaining_load=self._force_reference.measure(state[26:29],state[3:7])['load_change_norm_n']
                self._command_local(move_robot,hold,owner='transport_retreat_settle_hold',state=state,sim=sim)
                clear_streak=clear_streak+1 if remaining_load<=8. else 0
                if clear_streak>=3:
                    retreat=(matrix_from_xyz_xyzw(state[:7]),state,sim)
                    break
                if sim-start_sim>=1.:break
            if clear_streak<3:
                self._trace(event='transport_local_recovery_abort',sim_time=last,
                    reason='load_not_cleared_within_12mm',load_change_n=remaining_load,
                    geometry_input=False)
                return False
        retreat_pose,retreat_state,retreat_sim=retreat
        measured_retreat=float(np.dot(current[:3,3]-retreat_pose[:3,3],axial))
        if measured_retreat<.005 or measured_retreat>.014:
            self._trace(event='transport_local_recovery_abort',sim_time=retreat_sim,
                reason='retreat_outside_measured_bounds',measured_retreat_mm=measured_retreat*1000,
                geometry_input=False)
            return False
        amount=.008 if attempt==0 else .012
        shifted=None
        for side in range(2):
            if side:lateral=-lateral
            goal=retreat_pose.copy();goal[:3,3]+=lateral*amount
            self._trace(event='transport_lateral_side_started',sim_time=retreat_sim,
                side=side+1,axis_base=lateral.tolist(),requested_mm=amount*1000,
                geometry_input=False)
            shifted=self._move_local(get_observation,move_robot,goal,owner='transport_lateral_retry',
                max_duration_s=9.,maximum_load_change_n=10.,clear_load_change_n=8.,
                movement_axis=lateral,orthogonal_axis=axial,minimum_projected_m=.004)
            if shifted is not None:break
            if side==0:
                # Unload the first exploratory side before trying its
                # opposite. If the measured retreat hold is not regained,
                # stop instead of crossing a loaded obstacle.
                back=self._move_local(get_observation,move_robot,retreat_pose,
                    owner='transport_lateral_unload',max_duration_s=5.,
                    maximum_load_change_n=30.,clear_load_change_n=8.)
                if back is None:return False
        if shifted is None:return False
        final,state,sim=shifted
        displacement=final[:3,3]-retreat_pose[:3,3]
        lateral_m=float(np.dot(displacement,lateral))
        axial_drift=float(np.dot(displacement,axial))
        if lateral_m<.004 or abs(axial_drift)>.003:
            self._trace(event='transport_local_recovery_abort',sim_time=sim,
                reason='lateral_not_measured_or_axial_drift',measured_lateral_mm=lateral_m*1000,
                axial_drift_mm=axial_drift*1000,geometry_input=False)
            return False
        # Shift the remaining transport waypoints, never the final requested
        # port acquisition goal. Otherwise the resumed rule would immediately
        # drive back into the same blocked route.
        offset=lateral*lateral_m
        for index in range(self.index,len(self.stages)-1):
            name,xyz,rotation=self.stages[index]
            self.stages[index]=(name,np.asarray(xyz)+offset,rotation)
        self._resume_pose=final.copy();self._resume_sim_time=sim
        self._route_history=[final.copy()];self._route_forces=[float(np.linalg.norm(state[26:29]))]
        self._route_force_stop=False
        self._trace(event='transport_local_recovery_complete',sim_time=sim,
            attempt=attempt+1,measured_retreat_mm=measured_retreat*1000,
            measured_lateral_mm=lateral_m*1000,axial_drift_mm=axial_drift*1000,
            shifted_transport_stage_count=max(0,len(self.stages)-self.index-1),
            resumed_stage=self.stages[self.index][0],actor_owned=False,geometry_input=False)
        return True

    def _trace(self,**record):
        record['transport_attempt']=self._transport_attempt
        if self._record_route and record.get('event') in ('command','terminal_before_return') and 'state' in record:
            state=np.asarray(record['state'],float);pose=matrix_from_xyz_xyzw(state[:7])
            if not self._route_history or np.linalg.norm(pose[:3,3]-self._route_history[-1][:3,3])>=.002 or (
                    Rotation.from_matrix(pose[:3,:3]).inv()*Rotation.from_matrix(self._route_history[-1][:3,:3])).magnitude()>=np.deg2rad(2):
                self._route_history.append(pose)
                self._route_forces.append(float(np.linalg.norm(state[26:29])))
            if record.get('reason')=='force_stop':self._route_force_stop=True
        if self._record_route and record.get('event')=='rule_stage_completed' and record.get('stage')=='clear_start_side' and self._route_history:
            self._route_clear_checkpoint=len(self._route_history)-1
        super()._trace(**record)

    def _retreat_route(self,get_observation,move_robot,episode_start):
        if len(self._route_history)<3:return False
        checkpoint=self._route_clear_checkpoint or 0
        path=list(reversed(self._route_history[checkpoint:]));index=0;last_sim=-float('inf')
        clear_force=max(20.,self._route_forces[checkpoint]+8.)
        last_progress=None;best=float('inf');reference=None;compensation=np.zeros(3)
        started=time.monotonic();initial_force=None
        while time.monotonic()-started<90:
            obs=get_observation()
            if obs is None:time.sleep(.005);continue
            sim=self._control_sim_time(obs)
            if sim-episode_start>=155:return False
            if sim-last_sim<.05:time.sleep(.002);continue
            last_sim=sim;state=self.assembler.assemble_ros(obs);current=matrix_from_xyz_xyzw(state[:7])
            force=float(np.linalg.norm(state[26:29]))
            if initial_force is None:initial_force=force
            if force>max(60.,initial_force+10.):
                self._trace(event='transport_retreat_abort',sim_time=sim,reason='retreat_force_growth',force_norm_n=force)
                return False
            # Short lookahead can pass a2mm waypoint without entering its
            # tiny arrival sphere. Use actual monotonic progress in a10mm
            # tube around the recorded path, not the commanded reference.
            candidates=range(index,min(len(path),index+21))
            nearest=min(candidates,key=lambda j:np.linalg.norm(current[:3,3]-path[j][:3,3]))
            tube_error=float(np.linalg.norm(current[:3,3]-path[nearest][:3,3]))
            tube_angle=float((Rotation.from_matrix(current[:3,:3]).inv()*Rotation.from_matrix(path[nearest][:3,:3])).magnitude())
            if nearest>index and tube_error<=.010 and tube_angle<=np.deg2rad(3.):
                index=nearest;best=float('inf');last_progress=sim
            # Waypoints are actual past poses. Advance only after measured
            # arrival; no commanded-only checkpoint counts as a retreat.
            while index<len(path)-1 and np.linalg.norm(current[:3,3]-path[index][:3,3])<=.006:
                index+=1;best=float('inf');last_progress=sim
            target=path[index]
            error=float(np.linalg.norm(current[:3,3]-target[:3,3]))
            angle=float((Rotation.from_matrix(current[:3,:3]).inv()*Rotation.from_matrix(target[:3,:3])).magnitude())
            if index==len(path)-1 and error<=.006 and angle<=np.deg2rad(3) and force<=clear_force:
                self._trace(event='transport_measured_route_retreat_complete',sim_time=sim,
                    error_mm=error*1000,force_norm_n=force,measured_waypoints=len(path),checkpoint='measured_before_lateral_transfer',geometry_input=False)
                return True
            if error<best-.001:best=error;last_progress=sim
            if last_progress is not None and sim-last_progress>8:
                self._trace(event='transport_retreat_abort',sim_time=sim,reason='no_measured_reverse_progress')
                return False
            # Use a short lookahead on the RECORDED reverse path so compliant
            # load bias cannot trap the robot at every2mm history sample.
            goal=target.copy();look=index
            while look<len(path)-1 and np.linalg.norm(path[look][:3,3]-current[:3,3])<.020:look+=1
            goal=path[look].copy()
            compensation=np.clip(compensation+.05*.10*(target[:3,3]-current[:3,3]),-.012,.012)
            goal[:3,3]+=compensation
            if reference is None:reference=current.copy()
            d=float(np.linalg.norm(goal[:3,3]-reference[:3,3]));a=float((Rotation.from_matrix(reference[:3,:3]).inv()*Rotation.from_matrix(goal[:3,:3])).magnitude())
            alpha=min(1.,.05/max(d/.040,a/.15,.05))
            command=np.eye(4);command[:3,3]=reference[:3,3]+alpha*(goal[:3,3]-reference[:3,3])
            command[:3,:3]=Slerp([0.,1.],Rotation.from_matrix(np.stack([reference[:3,:3],goal[:3,:3]])))([alpha]).as_matrix()[0]
            reference=command.copy()
            if hasattr(self._parent_node,'_target_mode'):self._parent_node._target_mode=TargetMode.MODE_CARTESIAN
            stiffness,damping=self._cartesian_gains()
            self.set_pose_target(move_robot=move_robot,pose=build_pose_from_vectors(command[:3,3],Rotation.from_matrix(command[:3,:3]).as_quat()),
                frame_id='base_link',stiffness=stiffness,damping=damping)
            self._trace(event='transport_reverse_path_command',sim_time=sim,history_index=index,
                measured_state=state.tolist(),absolute_tcp_target=command.tolist(),force_norm_n=force,
                measured_path_tube_error_mm=tube_error*1000,actor_owned=False,geometry_input=False)
        return False

    def insert_cable(self,task,get_observation,move_robot,send_feedback,**kwargs):
        observation=get_observation()
        if observation is None:return False
        start=self._control_sim_time(observation)
        original=copy.deepcopy(self.spec['families']['sc']['bypass'])
        wall_start=time.monotonic();retained=[observation]
        def bounded_observation():
            value=get_observation()
            if value is not None:
                stamp=self._control_sim_time(value)
                if stamp<start or stamp-start>=155 or time.monotonic()-wall_start>=290:
                    raise TransportRecoveryBudget('whole_transport_budget')
                retained[0]=value
            return value
        try:
            self._transport_attempt=0;self._route_history=[];self._route_forces=[]
            self._route_clear_checkpoint=None;self._route_force_stop=False
            self._record_route=False;self._resume_transport=False
            self._force_reference=None;self._contact_monitor.reset()
            if not self._calibrate_transport_force(bounded_observation,move_robot):return False
            for attempt in range(3):
                self._transport_attempt=attempt;self._route_force_stop=False
                self._record_route=True;contact=False
                try:
                    super().insert_cable(task,bounded_observation,move_robot,send_feedback,**kwargs)
                except TransportLocalContact:
                    contact=True
                self._record_route=False
                if self.rule_final_valid:return False  # Transport only; never claim insertion.
                if not(contact or self._route_force_stop):return False
                if attempt>=2:
                    self._trace(event='transport_local_recovery_abort',reason='two_retry_budget_exhausted',
                        geometry_input=False)
                    return False
                if not self._recover_local(bounded_observation,move_robot,attempt=attempt):return False
                self._resume_transport=True
        except TransportRecoveryBudget:
            self.rule_final_valid=False
            self._trace(event='transport_recovery_budget_exhausted',geometry_input=False)
            return False
        finally:
            self._record_route=False;self._resume_transport=False
            self.spec['families']['sc']['bypass']=original
            # This happens before the outer action returns, including failed
            # retreats; never score an earlier intermediate terminal instead.
            current=retained[0]
            if current is not None:
                self._trace(event='terminal_before_return',reason='sample_transport_recovery_complete',
                    sim_time=self._control_sim_time(current),state=self.assembler.assemble_ros(current).tolist(),
                    observation_source='current_outer_action_before_return',geometry_input=False)
        return False
