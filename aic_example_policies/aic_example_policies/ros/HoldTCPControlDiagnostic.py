"""Fixed measured-TCP hold diagnostic. No learned actor, teacher or target geometry."""
import json,os,time
import numpy as np
from aic_model.policy import Policy
from lerobot_robot_aic.runtime_features import base_state_from_ros_observation


class HoldTCPControlDiagnostic(Policy):
    def insert_cable(self,task,get_observation,move_robot,send_feedback,**kwargs):
        stiffness=json.loads(os.environ.get('AIC_SHARED_PORT_TCP_CARTESIAN_STIFFNESS','[90,90,90,50,50,50]'))
        damping=json.loads(os.environ.get('AIC_SHARED_PORT_TCP_CARTESIAN_DAMPING','[50,50,50,20,20,20]'))
        path=os.environ['AIC_SHARED_PORT_TCP_TRACE'];target=None;first=None;last=None;wall=time.monotonic()
        while time.monotonic()-wall<45:
            obs=get_observation()
            if obs is None:time.sleep(.01);continue
            stamp=obs.controller_state.header.stamp;t=stamp.sec+stamp.nanosec*1e-9
            if t==last:time.sleep(.002);continue
            last=t;state=base_state_from_ros_observation(obs)
            if target is None:target=obs.controller_state.tcp_pose;first=t
            if t-first>=12:break
            if np.linalg.norm(state[26:29])>40:raise RuntimeError('Diagnostic raw force cap exceeded')
            self.set_pose_target(move_robot=move_robot,pose=target,frame_id='base_link',stiffness=stiffness,damping=damping)
            with open(path,'a') as f:
                f.write(json.dumps(dict(event='fixed_hold_diagnostic',sim_time=t,state=state.tolist(),
                    fixed_tcp_target=[target.position.x,target.position.y,target.position.z],
                    stiffness=stiffness,damping=damping,actor=False,geometry_input=False))+'\n')
        obs=get_observation()
        if obs is not None:
            with open(path,'a') as f:f.write(json.dumps(dict(event='terminal_before_return',state=base_state_from_ros_observation(obs).tolist()))+'\n')
        return True
