"""Privileged recorded-route interface diagnostic. NOT an autonomous policy."""
import bisect
import json
import os
import time
import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Pose
from aic_model.policy import Policy
from lerobot_robot_aic.runtime_features import base_state_from_ros_observation


class RecordedTeacherControlDiagnostic(Policy):
    def insert_cable(self,task,get_observation,move_robot,send_feedback,**kwargs):
        source=json.load(open(os.environ['AIC_TEACHER_REPLAY_SOURCE']))
        assert source['task_module']==task.target_module_name
        assert source['transport_only'] and not source['final_scene']
        rows=source['commands'];times=[r['elapsed_s'] for r in rows]
        cap=float(os.environ['AIC_TEACHER_REPLAY_CAP']);assert cap in (.02,.06)
        trace=os.environ['AIC_SHARED_PORT_TCP_TRACE'];first=None;last=-1.;wall=time.monotonic()
        while time.monotonic()-wall<180:
            obs=get_observation()
            if obs is None:time.sleep(.005);continue
            stamp=obs.controller_state.header.stamp;t=stamp.sec+stamp.nanosec*1e-9
            if first is None:first=t
            if t-first>times[-1]+8:break
            if t-last<.049:time.sleep(.002);continue
            last=t;state=base_state_from_ros_observation(obs)
            if np.linalg.norm(state[26:29])>40:
                # Preserve the causal terminal state before an exception allows
                # the engine to reset. Never fetch a post-reset replacement.
                with open(trace,'a') as f:f.write(json.dumps(dict(event='terminal_before_return',
                    sim_time=t,state=state.tolist(),privileged_recorded_route=True,
                    reason='diagnostic_raw_wrench_cap',aborted=True))+'\n')
                raise RuntimeError('Diagnostic raw wrench cap exceeded')
            row=rows[max(0,bisect.bisect_right(times,t-first)-1)]
            target=np.asarray(row['pose']);r=Rotation.from_quat(state[3:7])
            requested=r.inv().apply(target[:3]-state[:3]);executed=np.clip(requested,-cap,cap)
            xyz=np.asarray(state[:3])+r.apply(executed)
            pose=Pose();pose.position.x,pose.position.y,pose.position.z=map(float,xyz)
            pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w=map(float,target[3:])
            self.set_pose_target(move_robot=move_robot,pose=pose,frame_id='base_link',
                stiffness=[90,90,90,50,50,50],damping=[50,50,50,20,20,20])
            with open(trace,'a') as f:f.write(json.dumps(dict(event='recorded_teacher_command',sim_time=t,
                elapsed_s=t-first,state=state.tolist(),requested_body_translation=requested.tolist(),
                executed_body_translation=executed.tolist(),recorded_base_target=target.tolist(),
                capped=bool(np.any(abs(requested)>cap)),cap_m=cap,actor=False,
                privileged_recorded_route=True,live_geometry=False))+'\n')
        obs=get_observation()
        if obs is not None:
            with open(trace,'a') as f:f.write(json.dumps(dict(event='terminal_before_return',
                state=base_state_from_ros_observation(obs).tolist(),privileged_recorded_route=True))+'\n')
        return True
