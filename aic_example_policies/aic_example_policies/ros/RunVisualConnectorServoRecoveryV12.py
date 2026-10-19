"""Track a fixed, ramped lateral retry goal instead of integrating moving steps."""
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .RunVisualConnectorServoRecoveryV10 import RunVisualConnectorServoRecoveryV10
from .RunVisualConnectorServoRecoveryV5 import RunVisualConnectorServoRecoveryV5
from .RunRuleThenSharedRecovery import measured_pose
from shared_port_tcp_geometry import matrix_from_xyz_xyzw
from correlated_lateral_retry import lateral_retry_direction


class RunVisualConnectorServoRecoveryV12(RunVisualConnectorServoRecoveryV10):
    def _lateral_retry(self,get_observation,move_robot,*,route_level):
        observation=get_observation()
        if observation is None:return False
        state=self.local.assembler.assemble_ros(observation);start=matrix_from_xyz_xyzw(state[:7])
        inverse=np.linalg.inv(self.local.visual_port);body=self.local.visual_body_tip
        tip=(inverse@np.r_[start[:3,3]+start[:3,:3]@body,1.])[:3]
        if tip[2]>-.005:
            self._trace(event='recovery_abort',reason='lateral_retry_while_plug_may_be_inside');return False
        retreat=start[:3,3]-np.asarray(self.blocked_pose.xyz)
        direction=lateral_retry_direction(retreat,self.local.visual_port[:3,2],self.exploration_rng)
        amplitude=.010 if route_level else .003
        while amplitude>=.0005:
            candidate=tip+inverse[:3,:3]@(direction*amplitude)
            if np.linalg.norm(candidate[:2])<=.025 and candidate[2]<=-.005:break
            amplitude*=.5
        if amplitude<.0005 or (route_level and amplitude<.003):return False
        goal=start[:3,3]+direction*amplitude
        started=time.monotonic();first=None;last=None;arrival=0;integral=np.zeros(3)
        self.local.clear_pending_actions();self.recovery_command=None
        self._trace(event='lateral_retry_started',sampled_direction_base=direction.tolist(),amplitude_m=amplitude,
            fixed_measured_goal_base_m=goal.tolist(),integral_reset=True,geometry_input=False)
        while time.monotonic()-started<15 and time.monotonic()<self.episode_deadline-2:
            observation=get_observation()
            if observation is None:time.sleep(.005);continue
            sim=self.local._control_sim_time(observation)
            if sim-self.insertion_start_sim>self.total_budget_s:return False
            if last is not None and sim-last<.049:time.sleep(.002);continue
            dt=.05 if last is None else min(.1,sim-last);last=sim
            state=self.local.assembler.assemble_ros(observation);current=matrix_from_xyz_xyzw(state[:7]);force=self._force(state)
            if first is None:first=sim
            if force>=8 or np.linalg.norm(state[26:29])>40:
                self._trace(event='recovery_abort',reason='new_load_during_lateral_retry',load_change_n=force);return False
            self.supervisor._append_history(measured_pose(state),force,'supervisor_lateral_retry',self.route)
            travelled=min(amplitude,.005*(sim-first));reference=start[:3,3]+direction*travelled
            integral=np.clip(integral+.6*dt*(reference-state[:3]),-.005,.005)
            target=start.copy();target[:3,3]=reference+self.unloaded_reference_bias_base_m+integral
            # Bypass moving-step compensation: this loop already builds the
            # complete API reference for its fixed measured physical goal.
            RunVisualConnectorServoRecoveryV5._command(self,move_robot,target,owner='supervisor_lateral_exploration')
            distance=float(np.linalg.norm(goal-state[:3]))
            self._trace(event='lateral_retry_tracking',sim_time=sim,fixed_goal_base_m=goal.tolist(),
                nominal_reference_base_m=reference.tolist(),integral_base_m=integral.tolist(),
                measured_state=state.tolist(),remaining_mm=distance*1000,load_change_n=force,geometry_input=False)
            arrival=arrival+1 if travelled>=amplitude and distance<.00075 and np.linalg.norm(state[7:10])<.005 else 0
            if arrival>=3:
                moved=float(np.linalg.norm(state[:3]-start[:3,3]))
                if route_level and moved<.003:return False
                self.last_measured_lateral_retry_m=moved
                self._trace(event='lateral_retry_complete',sim_time=sim,route_level=route_level,
                    measured_displacement_m=(state[:3]-start[:3,3]).tolist(),sampled_direction_base=direction.tolist(),
                    amplitude_m=amplitude,seed=self.exploration_seed,actor_owned=False,
                    exploration_source='supervisor_coherent_direction_not_RL',geometry_input=False)
                if self.recovery_command is None:raise ValueError('Missing actual issued lateral reference')
                self.local.clear_pending_actions();self.local.governor=None
                self.local.governor_seed=(self.recovery_command.copy(),self.recovery_command_stamp)
                self.servo_stall_history=[];self.angle_retry+=1
                proposals=[(2.5,0,0),(-2.5,0,0),(0,2.5,0),(0,-2.5,0),(0,0,2.5),(0,0,-2.5)]
                if self.angle_retry>len(proposals):
                    self._trace(event='recovery_abort',reason='bounded_angular_retry_budget_exhausted');return False
                self.servo_goal_rotation=Rotation.from_rotvec(np.deg2rad(proposals[self.angle_retry-1])).as_matrix()
                self._trace(event='backoff_reference_reseeded',sim_time=self.recovery_command_stamp,
                    issued_tcp_target=self.recovery_command.tolist(),goal_rotation_offset_rotvec_deg=proposals[self.angle_retry-1],
                    exploration_source='fixed_symmetric_angles_after_observed_clearance_not_RL',geometry_input=False)
                return True
        self._trace(event='recovery_abort',reason='lateral_retry_timeout',sim_time=last);return False
