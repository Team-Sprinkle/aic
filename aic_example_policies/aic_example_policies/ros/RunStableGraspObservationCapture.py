"""Perception-only hold; retain terminal only after a quiet observation window."""
import json
import os
import time
from pathlib import Path
import numpy as np
from aic_model.policy import Policy, build_pose_from_vectors
from aic_control_interfaces.msg import TargetMode
from lerobot_robot_aic.runtime_features import base_state_from_ros_observation


class RunStableGraspObservationCapture(Policy):
    def insert_cable(self, task, get_observation, move_robot, send_feedback, **kwargs):
        started = time.monotonic()
        first = last = quiet = None
        pose = terminal = None
        while time.monotonic()-started < 100:
            observation = get_observation()
            if observation is None:
                time.sleep(.005)
                continue
            stamp = observation.controller_state.header.stamp
            sim = stamp.sec+stamp.nanosec*1e-9
            if last is not None and sim-last < .049:
                time.sleep(.002)
                continue
            state = base_state_from_ros_observation(observation)
            if first is None:
                first = sim
                pose = build_pose_from_vectors(state[:3], state[3:7])
                if hasattr(self._parent_node, '_target_mode'):
                    self._parent_node._target_mode = TargetMode.MODE_CARTESIAN
            last = sim
            stable = np.linalg.norm(state[7:10]) < .005 and np.linalg.norm(state[10:13]) < .05
            quiet = (sim if quiet is None else quiet) if stable else None
            self.set_pose_target(move_robot=move_robot, pose=pose, frame_id='base_link',
                                 stiffness=[90,90,90,50,50,50], damping=[50,50,50,20,20,20])
            # A freshly attached cable can produce a spawn impulse. Hold the
            # initial public pose during a bounded settling period; never
            # label that transient. Stable retained terminals still require
            # the same <40N bound. This is not an insertion force-guard change.
            if (sim-first >= 5 and np.linalg.norm(state[26:29]) >= 40) or sim-first > 20:
                break
            if sim-first >= 5 and quiet is not None and sim-quiet >= 2.7:
                terminal = dict(event='grasp_capture_terminal', sim_time=sim,
                    observation_wall_time_ns=time.time_ns(), state=state.tolist(),
                    geometry_input=False, pose_only=True, BC_expert=False,
                    quiet_window_s=sim-quiet, stable_terminal=True)
                break
            send_feedback('perception-only hold; waiting for quiet native image labels')
        if terminal is None:
            raise RuntimeError('No quiet perception hold within bounded time/force limits')
        with Path(os.environ['AIC_SHARED_PORT_TCP_TRACE']).open('a') as output:
            output.write(json.dumps(terminal)+'\n')
        return False
