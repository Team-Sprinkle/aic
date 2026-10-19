"""Teacher-only SFP near-port handoff to an observation-only local actor.

This is a diagnostic rollout, not an autonomous full-start policy: stock
CheatCode owns transport and reads simulator TF. The local actor receives
neither those transforms nor CheatCode targets.
"""
import os
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from rclpy.time import Time
from tf2_ros import TransformException

from .CheatCode import CheatCode


class _NearPortHandoff(Exception):
    pass


class CheatCodeThenLocal(CheatCode):
    def __init__(self,parent_node):
        super().__init__(parent_node)
        self._candidate=False;self._aligned_candidate=False;self._near_count=0
        self._handoff_offset=float(os.environ.get('AIC_SFP_LOCAL_HANDOFF_Z_OFFSET_M','.03'))
        self._handoff_hold=float(os.environ.get('AIC_SFP_LOCAL_HANDOFF_HOLD_SEC','.5'))
        self._physical_standoff=float(os.environ.get('AIC_SFP_LOCAL_HANDOFF_TIP_STANDOFF_M','0'))
        self._lateral_max=float(os.environ.get('AIC_SFP_LOCAL_HANDOFF_LATERAL_MAX_M','0'))
        self._axial_anchor=None
        self._axial_anchor_sim_time=None
        self._physical_count=0
        self._last_teacher_z_offset=None
        if self._physical_standoff and not .01<=self._physical_standoff<=.1:
            raise ValueError('Physical plug-tip handoff standoff must be 10–100 mm')
        if self._lateral_max and not .0005<=self._lateral_max<=.01:
            raise ValueError('Lateral handoff tolerance must be 0.5–10 mm')

    def calc_gripper_pose(self,port_transform,slerp_fraction=1.0,position_fraction=1.0,
                          z_offset=0.1,reset_xy_integrator=False):
        self._last_teacher_z_offset=float(z_offset)
        self._aligned_candidate=(position_fraction>=.999 and slerp_fraction>=.999)
        self._candidate=(position_fraction>=.999 and slerp_fraction>=.999 and
                         z_offset<=self._handoff_offset)
        return super().calc_gripper_pose(port_transform,slerp_fraction,
                                         position_fraction,z_offset,reset_xy_integrator)

    def _resume_teacher_insertion(self,task,move_robot):
        """Privileged stage-preserving handoff diagnostic, never deployable actor."""
        if self._last_teacher_z_offset is None:
            raise RuntimeError('No teacher descent stage to resume')
        self._teacher_resume=True
        audit=os.environ.get('AIC_TRANSPORT_EVAL_AUDIT')
        if audit:
            Path(audit,'teacher_takeover.json').write_text(json.dumps({
                'schema':'aic_privileged_stage_preserving_teacher_takeover/v1',
                'sim_time_s':self.time_now().nanoseconds/1e9,
                'actor_decision_limit':int(os.environ.get('AIC_SFP_LOCAL_DECISION_LIMIT','0')),
                'actor_generated_prefix':os.environ.get('AIC_SFP_STAGE_PRESERVE_AFTER_LOCAL','0')=='1',
                'privileged_teacher_after_takeover':True,
                'autonomous_success_claim':False},indent=2)+'\n')
        port=self._parent_node._tf_buffer.lookup_transform(
            'base_link',self._port_frame,Time()).transform
        start=float(self._last_teacher_z_offset)
        if not -.005<=start<=.02:
            raise ValueError(f'Teacher takeover outside local descent stage: {start}')
        self.get_logger().info(f'Stage-preserving privileged teacher from z_offset={start:.5f} m')
        for _ in range(20):
            if self._task_completed_in_simulation(task):return True
            self._send_delta_pose_target(move_robot,
                self.calc_gripper_pose(port,z_offset=start,reset_xy_integrator=True))
            self.sleep_for(.05)
        end=-.015
        steps=max(1,int(math.ceil(max(0.,start-end)/.0009/.05)))
        for step in range(1,steps+1):
            if self._task_completed_in_simulation(task):return True
            s=step/steps;fraction=10*s**3-15*s**4+6*s**5
            z=start-fraction*(start-end)
            self._send_delta_pose_target(move_robot,
                self.calc_gripper_pose(port,z_offset=z))
            self.sleep_for(.05)
        for _ in range(100):
            if self._task_completed_in_simulation(task):return True
            self.sleep_for(.05)
        return self._task_completed_in_simulation(task)

    def _send_delta_pose_target(self,move_robot,target_pose):
        if getattr(self,'_teacher_resume',False):
            return super()._send_delta_pose_target(move_robot,target_pose)
        if self._physical_standoff:
            try:
                port=self._parent_node._tf_buffer.lookup_transform('base_link',self._port_frame,Time()).transform
                tip=self._parent_node._tf_buffer.lookup_transform('base_link',self._tip_frame,Time()).transform
            except TransformException:
                self._physical_count=0
                super()._send_delta_pose_target(move_robot,target_pose)
                return
            center=np.array([port.translation.x,port.translation.y,port.translation.z])
            plug=np.array([tip.translation.x,tip.translation.y,tip.translation.z])
            orientation=port.rotation
            port_rotation=Rotation.from_quat([orientation.x,orientation.y,orientation.z,orientation.w])
            local=port_rotation.inv().apply(plug-center)
            lateral=float(np.linalg.norm(local[:2]))
            if self._lateral_max:
                if (self._aligned_candidate and local[2]>=-self._physical_standoff and
                    self._axial_anchor is None):
                    tcp=self._parent_node._tf_buffer.lookup_transform('base_link','gripper/tcp',Time()).transform
                    axis=port_rotation.apply([0.,0.,1.])
                    tcp_xyz=np.array([tcp.translation.x,tcp.translation.y,tcp.translation.z])
                    self._axial_anchor=float(np.dot(tcp_xyz-center,axis))
                    self._axial_anchor_sim_time=self.time_now().nanoseconds/1e9
                    self.get_logger().info(
                        f'Physical handoff axial hold begun; tip axial={local[2]:.5f} m, '
                        f'lateral={lateral:.5f} m')
                if self._axial_anchor is not None:
                    axis=port_rotation.apply([0.,0.,1.])
                    target=np.array([target_pose.position.x,target_pose.position.y,target_pose.position.z])
                    target+=axis*(self._axial_anchor-float(np.dot(target-center,axis)))
                    target_pose.position.x,target_pose.position.y,target_pose.position.z=map(float,target)
            super()._send_delta_pose_target(move_robot,target_pose)
            self._physical_count=(self._physical_count+1 if self._aligned_candidate and
                                  (self._axial_anchor is not None if self._lateral_max else
                                   local[2]>=-self._physical_standoff) and
                                  (not self._lateral_max or lateral<=self._lateral_max) else 0)
            if self._physical_count>=2:
                audit=os.environ.get('AIC_TRANSPORT_EVAL_AUDIT')
                if audit:
                    Path(audit,'physical_tip_handoff.json').write_text(json.dumps({
                        'schema':'aic_teacher_only_physical_tip_handoff/v1',
                        'measured_tip_in_port_link_m':local.tolist(),
                        'threshold_m':self._physical_standoff,
                        'lateral_tolerance_m':self._lateral_max or None,
                        'axial_hold_started_sim_time_s':self._axial_anchor_sim_time,
                        'sim_time_s':self.time_now().nanoseconds/1e9,
                        'true_geometry_actor_input':False},indent=2)+'\n')
                raise _NearPortHandoff
            return
        super()._send_delta_pose_target(move_robot,target_pose)
        self._near_count=self._near_count+1 if self._candidate else 0
        if self._near_count>=5:
            self.sleep_for(self._handoff_hold)
            raise _NearPortHandoff

    def insert_cable(self,task,get_observation,move_robot,send_feedback):
        if not task.target_module_name.startswith('nic_card_mount_'):
            raise ValueError('CheatCodeThenLocal is only for SFP-to-NIC diagnostics')
        self._port_frame=f'task_board/{task.target_module_name}/{task.port_name}_link'
        # ``plug_name`` denotes the module body. The contact tip is a distinct
        # link roughly 50 mm farther forward, so body-frame thresholds silently
        # hand control off after full insertion.
        self._tip_frame=f'{task.cable_name}/sfp_tip_link'
        # Load all actor and perception weights before teacher transport so
        # model initialization cannot leave the robot drifting at handoff.
        from .RunPredictedPortLocal import RunPredictedPortLocal
        local=RunPredictedPortLocal(self._parent_node,
            checkpoint_path=os.environ['AIC_OFFICIAL_TEACHER_LOCAL_ACTOR_CHECKPOINT'])
        self.get_logger().info('Local actor preloaded before teacher transport')
        try:
            result=super().insert_cable(task,get_observation,move_robot,send_feedback)
            if self._lateral_max:
                self.get_logger().error('Physical start generator never reached lateral handoff tolerance')
                return False
            return result
        except _NearPortHandoff:
            self.get_logger().info('Teacher-only SFP transport ended at near-port handoff')
            if os.environ.get('AIC_SFP_POST_HANDOFF_STAGE_PRESERVING_TEACHER','0')=='1':
                if os.environ.get('AIC_SFP_STAGE_PRESERVE_AFTER_LOCAL','0')=='1':
                    self.set_delta_pose_target_from_components(
                        move_robot=move_robot,delta_position_xyz=np.zeros(3),
                        delta_rotation_xyz=np.zeros(3))
                    local.insert_cable(task,get_observation,move_robot,send_feedback)
                    self.get_logger().info('Privileged teacher takeover after bounded local actor')
                return self._resume_teacher_insertion(task,move_robot)
            if os.environ.get('AIC_SFP_POST_HANDOFF_TEACHER_DIAGNOSTIC','0')=='1':
                # Privileged start-feasibility control only. This re-enters
                # stock CheatCode from the measured handoff state; it does not
                # evaluate the visual actor and cannot be a policy success.
                self._physical_standoff=0.;self._lateral_max=0.
                self._axial_anchor=None;self._physical_count=0
                self._teacher_resume=True
                self.get_logger().info('Privileged CheatCode restart after handoff diagnostic')
                return super().insert_cable(task,get_observation,move_robot,send_feedback)
            # The last teacher command otherwise keeps moving the servo during
            # the actor's model-loading delay, causing an insertion before BC
            # makes its first decision. Freeze at the *measured* TCP first.
            self.set_delta_pose_target_from_components(
                move_robot=move_robot,
                delta_position_xyz=np.zeros(3),
                delta_rotation_xyz=np.zeros(3),
            )
            self.get_logger().info('Handoff hold command sent before local actor initialization')
            result=local.insert_cable(task,get_observation,move_robot,send_feedback)
            return bool(result or self._task_completed_in_simulation(task))
