"""Bounded RGB/proprio approach to the near-port actor starting region.

Not an actor improvement: the approach is an explicit observation-only
controller. Exact simulator geometry, scoring events and scene config are
never read. The actor remains frozen and owns insertion.
"""
import time
import numpy as np
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation
from .RunGuardedRuleThenLockedTipLocal import RunGuardedRuleThenLockedTipLocal
from shared_port_tcp_geometry import matrix_from_xyz_xyzw


class RunVisualApproachThenLockedTipLocal(RunGuardedRuleThenLockedTipLocal):
    def prepare_local_start(self,task,get_observation,move_robot,send_feedback):
        port=self.local.visual_port
        body=self.local.visual_body_tip
        inverse=np.linalg.inv(port)
        start_wall=time.monotonic()
        initial_sim=last_sim=None
        integral=np.zeros(2)
        stable_since=None
        progress_since=None
        best_distance=float('inf')
        goal_point=np.array([0.,0.,-.030])
        while time.monotonic()-start_wall<90:
            observation=get_observation()
            if observation is None:
                time.sleep(.002)
                continue
            sim=self._control_sim_time(observation)
            if last_sim is not None and sim-last_sim<.049:
                time.sleep(.002)
                continue
            last_sim=sim
            state=self.assembler.assemble_ros(observation)
            measured=matrix_from_xyz_xyzw(state[:7])
            point=(inverse@measured@np.r_[body,1.])[:3]
            if initial_sim is None:
                initial_sim=sim
                initial_point=point.copy()
                fixed_rotation=measured[:3,:3].copy()
                duration=max(4.,np.linalg.norm(goal_point-initial_point)/.015)
                progress_since=sim
            elapsed=sim-initial_sim
            distance=float(np.linalg.norm(point-goal_point))
            if distance<best_distance-.001:
                best_distance=distance
                progress_since=sim
            if elapsed>35 or sim-progress_since>10 or np.linalg.norm(state[26:29])>40:
                self._trace(event='visual_approach_failed',sim_time=sim,
                    reason=('simulation_budget' if elapsed>35 else 'no_progress' if
                            sim-progress_since>10 else 'raw_force_guard'),
                    state=state.tolist(),raw_force_N=float(np.linalg.norm(state[26:29])),
                    estimated_tip_port_m=point.tolist(),
                    geometry_input=False)
                return False
            reference=initial_point+(goal_point-initial_point)*min(1.,elapsed/duration)
            integral=np.clip(integral+reference[:2]-point[:2],-.05,.05)
            commanded=reference.copy()
            commanded[:2]+=.15*integral
            goal=measured.copy()
            goal[:3,:3]=fixed_rotation
            goal[:3,3]=(port@np.r_[commanded,1.])[:3]-fixed_rotation@body
            delta=np.linalg.inv(measured)@goal
            translation=np.clip(delta[:3,3],-.015,.015)
            rotvec=Rotation.from_matrix(delta[:3,:3]).as_rotvec()
            norm=np.linalg.norm(rotvec)
            if norm>.1:
                rotvec*=.1/norm
            relative=np.eye(4)
            relative[:3,3]=translation
            relative[:3,:3]=Rotation.from_rotvec(rotvec).as_matrix()
            command=measured@relative
            pose=Pose()
            pose.position.x,pose.position.y,pose.position.z=map(float,command[:3,3])
            pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w=map(
                float,Rotation.from_matrix(command[:3,:3]).as_quat())
            self.set_pose_target(move_robot=move_robot,pose=pose,frame_id='base_link',
                stiffness=[90,90,90,50,50,50],damping=[50,50,50,20,20,20])
            self._trace(event='observation_only_approach',sim_time=sim,state=state.tolist(),
                estimated_tip_port_m=point.tolist(),reference_tip_port_m=reference.tolist(),
                commanded_tip_port_m=commanded.tolist(),source='RGB_lock_and_measured_TCP',
                true_geometry_input=False,learned_actor_action=False,backoff_enabled=False)
            ready=(elapsed>=duration and abs(point[2]+.030)<=.0015
                   and np.linalg.norm(point[:2])<=.0005 and np.linalg.norm(state[7:10])<=.005)
            stable_since=(sim if stable_since is None else stable_since) if ready else None
            if stable_since is not None and sim-stable_since>=1:
                self._trace(event='observation_approach_actor_start',sim_time=sim,
                    estimated_tip_port_m=point.tolist(),state=state.tolist(),
                    geometry_input=False,actor_checkpoint_unchanged=True)
                return True
        return False
