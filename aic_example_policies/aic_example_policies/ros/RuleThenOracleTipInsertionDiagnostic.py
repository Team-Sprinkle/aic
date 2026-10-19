"""Privileged one-reset controller diagnosis. Never an autonomous policy.

Transport/handoff remain observation-only. ONLY this diagnostic then reads
simulator port/plug TF to establish whether the current controller can reach
full depth. No learned policy, reward or promotion consumes its targets.
"""
import json
import time
import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Pose
from rclpy.time import Time
from .RunYawMemoryTrackedRuleTransport import RunYawMemoryTrackedRuleTransport
from shared_port_tcp_geometry import matrix_from_xyz_xyzw


class RuleThenOracleTipInsertionDiagnostic(RunYawMemoryTrackedRuleTransport):
    def insert_cable(self,task,get_observation,move_robot,send_feedback,**kwargs):
        import os
        super().insert_cable(task,get_observation,move_robot,send_feedback,**kwargs)
        if not self.rule_final_valid or self.rule_failure:return False
        return self.run_oracle_local(task,get_observation,move_robot,send_feedback,**kwargs)

    def run_oracle_local(self,task,get_observation,move_robot,send_feedback,**kwargs):
        import os
        requested=int(np.argmax(self.task_vector(task)[2:4]))
        def tf(frame,when=None):
            m=self._parent_node._tf_buffer.lookup_transform('base_link',frame,Time() if when is None else when).transform
            return matrix_from_xyz_xyzw([m.translation.x,m.translation.y,m.translation.z,
                m.rotation.x,m.rotation.y,m.rotation.z,m.rotation.w])
        observation=get_observation()
        if observation is None:return False
        state=self.assembler.assemble_ros(observation);initial_sim=self._control_sim_time(observation)
        port=tf(f'task_board/sc_port_{requested}/sc_port_base_link_entrance')
        causal_calibration=os.environ.get('AIC_ORACLE_CAUSAL_TF_CALIBRATION','0')=='1'
        if causal_calibration:
            stamp=self._parent_node._tf_buffer.lookup_transform('base_link','gripper/tcp',Time()).header.stamp
            calibration_time=Time.from_msg(stamp)
            # Request both rigid frames at the same epoch. Wait for the tip
            # publisher if its latest sample precedes the robot sample.
            from rclpy.duration import Duration
            self._parent_node._tf_buffer.lookup_transform('base_link','cable_1/sc_tip_link',calibration_time,timeout=Duration(seconds=1.))
            tip=tf('cable_1/sc_tip_link',calibration_time)
            body=np.linalg.inv(tf('gripper/tcp',calibration_time))@tip
        else:
            tip=tf('cable_1/sc_tip_link')
            body=np.linalg.inv(matrix_from_xyz_xyzw(state[:7]))@tip
        current=np.linalg.inv(port)@tip
        prior=json.load(open(os.environ['AIC_ORACLE_MATING_PRIOR']))
        mating=Rotation.from_quat(prior['tip_orientation_port_xyzw']).as_matrix()
        target_rotation=port[:3,:3]@mating@body[:3,:3].T
        start_z=float(current[2,3]);approach_duration=max(4.,abs(-.025-start_z)/.015)
        use_integral=os.environ.get('AIC_ORACLE_LATERAL_INTEGRAL','0')=='1'
        axial_rate=float(os.environ.get('AIC_ORACLE_AXIAL_RATE_MPS','.001'))
        goal_depth=float(os.environ.get('AIC_ORACLE_GOAL_DEPTH_M','.01564'))
        if not .01564<=goal_depth<=.021:raise ValueError('Command depth outside recorded-teacher diagnostic range')
        if not 0<axial_rate<=.002:raise ValueError('Bounded axial rate diagnostic')
        integral=np.zeros(2)
        windup=float(os.environ.get('AIC_ORACLE_INTEGRAL_LIMIT_M','.05'))
        if not .05<=windup<=.08:raise ValueError('Integral outside bounded diagnosis')
        alignment_gate=os.environ.get('AIC_ORACLE_ALIGNMENT_GATE','0')=='1'
        alignment_lateral=float(os.environ.get('AIC_ORACLE_ALIGNMENT_LATERAL_M','.0005'))
        if not .0001<=alignment_lateral<=.0005:raise ValueError('Alignment tolerance outside diagnostic bounds')
        insertion_start=None
        stop_on_gate=os.environ.get('AIC_ORACLE_STOP_ON_PHYSICAL_GATE','0')=='1'
        gate_since=None
        last=-1.;wall=time.monotonic();terminal='bounded_runtime';final_state=state;final_sim=initial_sim
        self._trace(event='oracle_local_switch',sim_time=initial_sim,state=state.tolist(),
            privileged_geometry=True,autonomous_policy=False,body_tip_transform=body.tolist(),
            port_transform=port.tolist(),mating_prior=prior,physical_depth_gate_mm=14.,
            lateral_integral=use_integral,axial_rate_mps=axial_rate,integral_limit_m=windup,alignment_gate=alignment_gate,alignment_lateral_m=alignment_lateral,
            body_calibration_same_tf_epoch=causal_calibration,commanded_tip_depth_cap_m=goal_depth)
        while time.monotonic()-wall<180:
            observation=get_observation()
            if observation is None:time.sleep(.005);continue
            sim=self._control_sim_time(observation)
            if sim-last<.049:time.sleep(.002);continue
            last=sim;elapsed=sim-initial_sim;state=self.assembler.assemble_ros(observation)
            if alignment_gate and elapsed>=approach_duration+2 and insertion_start is None:
                alignment_tip=np.linalg.inv(port)@tf('cable_1/sc_tip_link')
                aligned=(np.linalg.norm(alignment_tip[:2,3])<=alignment_lateral and
                    (Rotation.from_matrix(mating).inv()*Rotation.from_matrix(alignment_tip[:3,:3])).magnitude()<np.pi/180)
                if aligned:insertion_start=elapsed
                elif elapsed>approach_duration+22:
                    final_state=state;final_sim=sim;terminal='oracle_alignment_timeout';break
            final_state=state;final_sim=sim
            if elapsed>70:break
            if np.linalg.norm(state[26:29])>40:terminal='raw_force_guard';break
            if elapsed<approach_duration:
                fraction=min(1.,elapsed/approach_duration)
                point=current[:3,3]*(1-fraction)+np.array([0.,0.,-.025])*fraction
            elif elapsed<approach_duration+2 or (alignment_gate and insertion_start is None):
                point=np.array([0.,0.,-.025])
            else:
                point=np.array([0.,0.,min(goal_depth,-.025+axial_rate*(elapsed-(insertion_start if alignment_gate else approach_duration+2)))])
            current_true_tip=np.linalg.inv(port)@tf('cable_1/sc_tip_link')
            if use_integral:
                integral=np.clip(integral+point[:2]-current_true_tip[:2,3],-windup,windup)
            commanded_point=point.copy();commanded_point[:2]+=.15*integral
            goal=np.eye(4);goal[:3,:3]=target_rotation
            goal[:3,3]=(port@np.r_[commanded_point,1.])[:3]-target_rotation@body[:3,3]
            measured=matrix_from_xyz_xyzw(state[:7]);delta=np.linalg.inv(measured)@goal
            translation=np.clip(delta[:3,3],-.015,.015)
            rotation=Rotation.from_matrix(delta[:3,:3]).as_rotvec();norm=np.linalg.norm(rotation)
            if norm>.1:rotation*=.1/norm
            relative=np.eye(4);relative[:3,3]=translation;relative[:3,:3]=Rotation.from_rotvec(rotation).as_matrix();command=measured@relative
            pose=Pose();pose.position.x,pose.position.y,pose.position.z=map(float,command[:3,3]);pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w=map(float,Rotation.from_matrix(command[:3,:3]).as_quat())
            self.set_pose_target(move_robot=move_robot,pose=pose,frame_id='base_link',
                stiffness=[90,90,90,50,50,50],damping=[50,50,50,20,20,20])
            measured_tip=np.linalg.inv(port)@tf('cable_1/sc_tip_link')
            self._trace(event='oracle_local_command',sim_time=sim,state=state.tolist(),
                requested_tip_point_port_m=point.tolist(),
                integral_correction_port_m=(.15*integral).tolist(),
                commanded_tip_point_port_m=commanded_point.tolist(),measured_tip_point_port_m=measured_tip[:3,3].tolist(),
                measured_tip_orientation_error_deg=float((Rotation.from_matrix(mating).inv()*Rotation.from_matrix(measured_tip[:3,:3])).magnitude()*180/np.pi),
                command_base=command.tolist(),privileged_geometry=True,autonomous_policy=False)
            if stop_on_gate:
                physical_ok=(measured_tip[2,3]>=.014 and np.linalg.norm(measured_tip[:2,3])<=.0005
                    and (Rotation.from_matrix(mating).inv()*Rotation.from_matrix(measured_tip[:3,:3])).magnitude()<np.pi/180)
                gate_since=(sim if gate_since is None else gate_since) if physical_ok else None
                if gate_since is not None and sim-gate_since>=2:
                    terminal='oracle_physical_gate';break
        self._trace(event='terminal_before_return',sim_time=final_sim,state=final_state.tolist(),
            reason=terminal,privileged_geometry=True,autonomous_policy=False)
        return terminal in ('bounded_runtime','oracle_physical_gate')
