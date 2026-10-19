"""Five-second public-pose hold for perception supervision only; no expert claim."""
import json,os,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from aic_model.policy import Policy,build_pose_from_vectors
from aic_control_interfaces.msg import TargetMode
from lerobot_robot_aic.runtime_features import base_state_from_ros_observation


class RunGraspObservationCapture(Policy):
    def insert_cable(self,task,get_observation,move_robot,send_feedback,**kwargs):
        started=time.monotonic();first=None;last=None;pose=None;terminal=None
        while time.monotonic()-started<30:
            obs=get_observation()
            if obs is None:time.sleep(.005);continue
            stamp=obs.controller_state.header.stamp;sim=stamp.sec+stamp.nanosec*1e-9
            if last is not None and sim-last<.049:time.sleep(.002);continue
            state=base_state_from_ros_observation(obs)
            if first is None:
                first=sim;pose=build_pose_from_vectors(state[:3],state[3:7])
                if hasattr(self._parent_node,'_target_mode'):self._parent_node._target_mode=TargetMode.MODE_CARTESIAN
            last=sim;terminal=dict(sim_time=sim,observation_wall_time_ns=time.time_ns(),state=state.tolist())
            if sim-first>=5:break
            self.set_pose_target(move_robot=move_robot,pose=pose,frame_id='base_link',stiffness=[90,90,90,50,50,50],damping=[50,50,50,20,20,20])
            send_feedback('perception-only free-grasp observation hold')
        if terminal is None:raise RuntimeError('No public controller observation')
        path=Path(os.environ['AIC_SHARED_PORT_TCP_TRACE'])
        with path.open('a') as output:output.write(json.dumps(dict(event='grasp_capture_terminal',**terminal,geometry_input=False,pose_only=True,BC_expert=False))+'\n')
        return False
