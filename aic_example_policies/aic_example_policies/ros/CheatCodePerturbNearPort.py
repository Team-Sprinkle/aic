"""Privileged training-data diagnostic: perturb a near-port pose, then recover.

This class is never a deployable actor. The unmodified CheatCode resumes after
the bounded perturbation so scored success determines whether the recovery
targets are useful supervised labels.
"""
import copy
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .CheatCode import CheatCode


class CheatCodePerturbNearPort(CheatCode):
    def __init__(self,parent_node):
        super().__init__(parent_node)
        self._applied=False
        self._ready=False
        self._port_rotation=None
        self._sign=float(os.environ.get('AIC_NEAR_PERTURB_SIGN','1'))
        self._lateral_m=float(os.environ.get('AIC_NEAR_PERTURB_LATERAL_M','.003'))
        self._yaw_deg=float(os.environ.get('AIC_NEAR_PERTURB_YAW_DEG','4'))
        self._hold_s=float(os.environ.get('AIC_NEAR_PERTURB_HOLD_S','.75'))
        if self._sign not in (-1.,1.) or not 0<self._lateral_m<=.005 or not 0<=self._yaw_deg<=7:
            raise ValueError('Perturbation outside bounded training-data pilot')

    def calc_gripper_pose(self,port_transform,slerp_fraction=1.,position_fraction=1.,
                          z_offset=.1,reset_xy_integrator=False):
        pose=super().calc_gripper_pose(port_transform,slerp_fraction,
                                       position_fraction,z_offset,reset_xy_integrator)
        if not self._applied and position_fraction>=.999 and slerp_fraction>=.999 and z_offset<=.03:
            q=port_transform.rotation
            self._port_rotation=Rotation.from_quat([q.x,q.y,q.z,q.w])
            self._ready=True
        return pose

    def _send_delta_pose_target(self,move_robot,target_pose):
        if self._ready and not self._applied:
            self._applied=True
            perturbed=copy.deepcopy(target_pose)
            offset=self._port_rotation.apply([self._sign*self._lateral_m,0.,0.])
            perturbed.position.x+=float(offset[0]);perturbed.position.y+=float(offset[1])
            perturbed.position.z+=float(offset[2])
            q=target_pose.orientation
            rotation=Rotation.from_quat([q.x,q.y,q.z,q.w])*Rotation.from_rotvec(
                [0.,0.,np.deg2rad(self._sign*self._yaw_deg)])
            quat=rotation.as_quat()
            perturbed.orientation.x=float(quat[0]);perturbed.orientation.y=float(quat[1])
            perturbed.orientation.z=float(quat[2]);perturbed.orientation.w=float(quat[3])
            start=self.time_now().nanoseconds
            for _ in range(max(1,int(round(self._hold_s/.05)))):
                super()._send_delta_pose_target(move_robot,perturbed)
                self.sleep_for(.05)
            end=self.time_now().nanoseconds
            log=Path('/audit/near_port_perturbation.json')
            log.write_text(json.dumps({'schema':'aic_privileged_near_port_perturbation/v1',
                'sim_start_ns':start,'sim_release_ns':end,'sign':self._sign,
                'lateral_m':self._lateral_m,'yaw_deg':self._yaw_deg,'hold_s':self._hold_s,
                'training_data_diagnostic_only':True,'true_geometry_deployable':False},indent=2)+'\n')
            self.get_logger().info(f'Near-port perturbation released at {end/1e9:.3f}s')
            return
        super()._send_delta_pose_target(move_robot,target_pose)
