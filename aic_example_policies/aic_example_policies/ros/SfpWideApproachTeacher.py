"""Privileged SFP demonstrations with explicitly excluded wide-start resets.

CheatCode supplies the successful SFP transport style. After transport, reset
to a measured 90/150/180mm plug approach and teach alignment/full insertion.
This collector is never an autonomous policy or an evaluation contender.
"""
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from rclpy.time import Time
from geometry_msgs.msg import Point, Pose, Quaternion, WrenchStamped, Wrench, Vector3
from std_msgs.msg import Header
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from .CheatCode import CheatCode


def matrix(transform):
    m=np.eye(4);q=transform.rotation;t=transform.translation
    m[:3,:3]=Rotation.from_quat([q.x,q.y,q.z,q.w]).as_matrix();m[:3,3]=[t.x,t.y,t.z]
    return m


def pose(matrix):
    q=Rotation.from_matrix(matrix[:3,:3]).as_quat()
    return Pose(position=Point(x=float(matrix[0,3]),y=float(matrix[1,3]),z=float(matrix[2,3])),
                orientation=Quaternion(x=float(q[0]),y=float(q[1]),z=float(q[2]),w=float(q[3])))


class SfpWideApproachTeacher(CheatCode):
    def __init__(self,parent_node):
        super().__init__(parent_node)
        self.wide=json.loads(os.environ['AIC_SFP_WIDE_APPROACH_JSON'])
        self.trace=Path(os.environ['AIC_SHARED_PORT_TCP_TRACE']);self.latest_force=None
        self.force_sub=parent_node.create_subscription(WrenchStamped,'/fts_broadcaster/wrench',self._force,10)
        axial=float(self.wide['axial_m']);lateral=np.asarray(self.wide['lateral_xy_m'])
        rv=np.deg2rad(self.wide['rotation_xyz_deg'])
        if not (.03<=axial<=.18 and np.linalg.norm(lateral)<=.025001 and np.linalg.norm(rv)<=np.deg2rad(15)):
            raise ValueError('Teacher start outside approved buffered envelope')

    def _force(self,message):
        f=message.wrench.force;self.latest_force=np.asarray([f.x,f.y,f.z])

    def log(self,event,**fields):
        row=dict(event=event,time_sec=self.time_now().nanoseconds/1e9,
                 privileged_geometry=True,autonomous_policy=False,**fields)
        with self.trace.open('a') as f:f.write(json.dumps(row)+'\n')

    def send(self,move_robot,target):
        update=MotionUpdate(header=Header(frame_id='base_link',stamp=self._parent_node.get_clock().now().to_msg()),
            pose=pose(target),target_stiffness=np.diag([90.,90.,90.,50.,50.,50.]).flatten().tolist(),
            target_damping=np.diag([50.,50.,50.,20.,20.,20.]).flatten().tolist(),
            feedforward_wrench_at_tip=Wrench(force=Vector3(),torque=Vector3()),
            wrench_feedback_gains_at_tip=[.5,.5,.5,0.,0.,0.],
            trajectory_generation_mode=TrajectoryGenerationMode(mode=TrajectoryGenerationMode.MODE_POSITION))
        move_robot(motion_update=update)

    def tf(self,frame,when=None):
        return self._parent_node._tf_buffer.lookup_transform('base_link',frame,Time() if when is None else when)

    def insert_cable(self,task,get_observation,move_robot,send_feedback):
        if not str(task.plug_name).startswith('sfp'):raise ValueError('SFP teacher only')
        self._task=task;self._latest_insertion_event_namespace=''
        port_frame=f'task_board/{task.target_module_name}/{task.port_name}_link'
        tip_frame=f'{task.cable_name}/{task.plug_name}_link'
        for frame in (port_frame,tip_frame,'gripper/tcp'):
            if not self._wait_for_tf('base_link',frame):return False
        transport_port=self.tf(port_frame).transform
        self.log('wide_teacher_transport_started',command_owner='transport_teacher',exclude_from_BC=True,
                 route='original SFP CheatCode interpolation to200mm above port; no SC route')
        for i in range(111):
            target=self.calc_gripper_pose(transport_port,slerp_fraction=i/110,position_fraction=i/110,z_offset=.2,reset_xy_integrator=True)
            q=target.orientation;p=target.position;m=np.eye(4);m[:3,:3]=Rotation.from_quat([q.x,q.y,q.z,q.w]).as_matrix();m[:3,3]=[p.x,p.y,p.z]
            self.send(move_robot,m);self.sleep_for(.05)
        # Physical SFP opening calibration is relative to NIC mount, not the
        # ambiguous SDF entrance frame. This access is collector-only.
        calibration=json.loads(Path('configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json').read_text())
        port_index=str(task.port_name.rsplit('_',1)[1]);offset=calibration['nic_mount_to_port_opening'][port_index]
        fixed=np.eye(4);fixed[:3,:3]=Rotation.from_quat(offset['quat_xyzw']).as_matrix();fixed[:3,3]=offset['xyz_m']
        port=matrix(self.tf(f'task_board/{task.target_module_name}').transform)@fixed
        a=self.tf('gripper/tcp');b=self.tf(tip_frame)
        stamp=min(a.header.stamp.sec*10**9+a.header.stamp.nanosec,b.header.stamp.sec*10**9+b.header.stamp.nanosec)
        when=Time(nanoseconds=stamp);tcp=matrix(self.tf('gripper/tcp',when).transform);body=np.linalg.inv(tcp)@matrix(self.tf(tip_frame,when).transform)
        desired=np.eye(4);desired[:3,:3]=Rotation.from_rotvec(np.deg2rad(self.wide['rotation_xyz_deg'])).as_matrix()
        desired[:3,3]=[*self.wide['lateral_xy_m'],-float(self.wide['axial_m'])]
        goal=port@desired@np.linalg.inv(body)
        self.log('supervisor_approach_reset_started',command_owner='dataset_reset_supervisor',exclude_from_BC=True,
                 requested=self.wide,tcp_tip_transform=body.tolist(),base_port_transform=port.tolist(),calibration_stamp_ns=stamp)
        duration=max(np.linalg.norm(goal[:3,3]-tcp[:3,3])/.04,
                     (Rotation.from_matrix(tcp[:3,:3]).inv()*Rotation.from_matrix(goal[:3,:3])).magnitude()/.3,1.)
        if duration>12:raise ValueError('Wide SFP reset exceeds local reset envelope')
        steps=int(math.ceil(duration/.05));rotations=Slerp([0,1],Rotation.from_matrix(np.stack([tcp[:3,:3],goal[:3,:3]])))
        for i in range(steps+1):
            s=i/steps;alpha=10*s**3-15*s**4+6*s**5;target=np.eye(4)
            target[:3,3]=tcp[:3,3]+alpha*(goal[:3,3]-tcp[:3,3]);target[:3,:3]=rotations([alpha]).as_matrix()[0]
            self.send(move_robot,target);self.sleep_for(.05)
        compensation=np.zeros(3);settled=0;previous=None
        for iteration in range(300):
            measured=matrix(self.tf('gripper/tcp').transform);error=goal[:3,3]-measured[:3,3]
            angle=(Rotation.from_matrix(measured[:3,:3]).inv()*Rotation.from_matrix(goal[:3,:3])).magnitude()
            speed=float('inf') if previous is None else np.linalg.norm(measured[:3,3]-previous)/.05;previous=measured[:3,3].copy()
            settled=settled+1 if np.linalg.norm(error)<.003 and angle<np.deg2rad(3) and speed<.004 else 0
            if settled>=3:break
            compensation=np.clip(compensation+.035*error,-.030,.030);target=goal.copy();target[:3,3]+=compensation
            self.send(move_robot,target);self.sleep_for(.05)
        actual=np.linalg.inv(port)@matrix(self.tf(tip_frame).transform);reset_error=float(np.linalg.norm(actual[:3,3]-desired[:3,3]))
        self.log('supervisor_approach_reset_released',command_owner='teacher',actual_tip_position_port_m=actual[:3,3].tolist(),
                 reset_tip_error_mm=reset_error*1000,reset_passed=reset_error<=.005)
        if reset_error>.005:
            self.terminal(task,port,tip_frame,False,'wide_reset_not_reached');return False
        return self.local(task,move_robot,port,body,compensation,tip_frame)

    def terminal(self,task,port,tip_frame,success,reason):
        observed=self.tf(tip_frame);relative=np.linalg.inv(port)@matrix(observed.transform)
        self.log('wide_teacher_terminal_before_return',success=bool(success),reason=reason,
                 measured_tip_port_m=relative[:3,3].tolist(),lateral_mm=float(np.linalg.norm(relative[:2,3])*1000),
                 orientation_error_deg=float(np.degrees(Rotation.from_matrix(relative[:3,:3]).magnitude())),
                 geometry_stamp_ns=observed.header.stamp.sec*10**9+observed.header.stamp.nanosec,
                 observation_wall_time_ns=time.time_ns(),simulator_event_acknowledged=self._task_completed_in_simulation(task))

    def local(self,task,move_robot,port,body,compensation,tip_frame):
        current=np.linalg.inv(port)@matrix(self.tf(tip_frame).transform);desired=current[:3,3].copy();rotation=Rotation.from_matrix(current[:3,:3])
        bias=port[:3,:3].T@compensation;stage='alignment';aligned_since=None;baseline_rows=[];baseline=None
        start=self.time_now().nanoseconds/1e9;last_z=float(desired[2]);last_progress=start;success=False;reason='teacher_time_limit'
        self.log('local_preinsert_align_started',command_owner='teacher',full_150mm_teacher=True,start_tip_port_m=desired.tolist())
        for iteration in range(2000):
            now=self.time_now().nanoseconds/1e9
            if now-start>100:break
            measured=np.linalg.inv(port)@matrix(self.tf(tip_frame).transform);point=measured[:3,3]
            angle=float(np.degrees(Rotation.from_matrix(measured[:3,:3]).magnitude()));lateral=float(np.linalg.norm(point[:2]))
            force=None if self.latest_force is None else self.latest_force.copy();force_delta=None
            if force is not None and np.linalg.norm(force)>40:reason='teacher_raw_force_guard';break
            if baseline is not None and force is not None:force_delta=float(np.linalg.norm(force-baseline))
            desired[:2]+=np.clip(-desired[:2],-.001,.001);rv=rotation.as_rotvec();norm=np.linalg.norm(rv)
            rotation=Rotation.from_rotvec(rv*max(0.,1-.0125/max(norm,1e-12)))
            aligned=lateral<=.0003 and angle<=.75
            aligned_since=(now if aligned_since is None else aligned_since) if aligned else None
            if stage=='alignment' and aligned_since is not None and now-aligned_since>=.5:stage='approach'
            if stage=='approach':
                if lateral<=.001 and angle<=1.5:desired[2]=min(-.025,desired[2]+.008*.05)
                if point[2]>=-.026 and aligned and force is not None:baseline_rows.append(force)
                if len(baseline_rows)>=10:baseline=np.median(baseline_rows,axis=0);stage='insertion';desired[2]=-.025
            if stage=='insertion':
                if lateral<=.0005 and angle<=1:desired[2]=min(.0469,desired[2]+.004*.05)
                if self._task_completed_in_simulation(task):
                    success=bool(point[2]>=.044 and lateral<=.0005 and angle<=1)
                    reason='teacher_full_physical_depth_lateral_orientation' if success else 'teacher_sim_completion_before_physical_gate'
                    self.log('wide_teacher_completion_observed',command_owner='teacher',success=success,
                             measured_tip_port_m=point.tolist(),further_motion_commands_disabled=True)
                    self.sleep_for(1.);break
                if point[2]>last_z+.00025:last_progress=now;last_z=float(point[2])
                if force_delta is not None and force_delta>=5 and now-last_progress>1:reason='teacher_blocked_contact';break
            bias=np.clip(bias+.02*(desired-point),-.025,.025);target=np.eye(4)
            target[:3,3]=desired+bias;target[:3,:3]=rotation.as_matrix();goal=port@target@np.linalg.inv(body)
            self.send(move_robot,goal)
            self.log('wide_teacher_local_command',command_owner='teacher',stage=stage,desired_tip_port_m=desired.tolist(),
                     measured_tip_port_m=point.tolist(),orientation_error_deg=angle,command_base=goal.tolist(),force_delta_n=force_delta)
            self.sleep_for(.05)
        self.terminal(task,port,tip_frame,success,reason)
        return success or self._task_completed_in_simulation(task)
