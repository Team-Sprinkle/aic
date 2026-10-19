"""Bounded observation-only docking after a learned SC transport proposal.

This is an explicit visual controller comparison, not improved BC action
prediction. Simulator geometry never enters the controller. Final acceptance
uses the unchanged requested-port verifier and two-second stationary gate.
"""
import time

import numpy as np
from scipy.spatial.transform import Rotation

from aic_control_interfaces.msg import TargetMode
from .RunStableHandoffTransport import RunStableHandoffTransport
from aic_model.policy import build_pose_from_vectors


class RunVisualDockingTransportV2(RunStableHandoffTransport):
    def _inspect_transport_handoff(self, observed, state, task_vector,
                                   get_observation, move_robot, decision_sim):
        if task_vector[0] == 1 or self.sc_shadow_verifier is None:
            raise ValueError('SC-only bounded docking comparison')
        requested = int(np.argmax(task_vector[2:4]))
        origin = np.asarray(state[:7], float)
        origin_rot = Rotation.from_quat(origin[3:7])
        reference_position = origin[:3].copy()
        reference_rotation = origin_rot
        last_sim = decision_sim
        valid_start = None
        previous = None
        agreeing = 0
        deadline = time.monotonic() + 45.
        while time.monotonic() < deadline:
            observation = get_observation()
            if observation is None:
                time.sleep(.005)
                continue
            sim = self._control_sim_time(observation)
            if sim-last_sim < .15:
                time.sleep(.005)
                continue
            last_sim = sim
            fresh = self.assembler.assemble_ros(observation)
            if np.linalg.norm(fresh[26:29]) > self.diagnostic_force_stop_n:
                self._trace(event='visual_docking_rejected',sim_time=sim,reason='force_stop',
                            force_norm_n=float(np.linalg.norm(fresh[26:29])))
                return False
            images = {c:self._shadow_raw_rgb(getattr(observation,c+'_image'))
                      for c in ('center','left','right')}
            report = self.sc_shadow_verifier.inspect(images,fresh,requested)
            speed = float(np.linalg.norm(fresh[7:10])*1000)
            angular = float(np.linalg.norm(fresh[10:13])*180/np.pi)
            self._trace(event='visual_docking_inspection',sim_time=sim,state=fresh.tolist(),
                        report=report,requested_port=requested,geometry_input=False,
                        tcp_speed_mm_s=speed,angular_speed_deg_s=angular)
            if report['reason'] not in ('not_at_selected_standoff','wrong_tcp_posture',
                                        'clear_selected_port_and_safe_standoff'):
                self._trace(event='visual_docking_rejected',sim_time=sim,reason=report['reason'])
                return False
            position = np.asarray(report['predicted_opening_base_m']) + self.sc_shadow_verifier.standoff_offset
            rotation = (Rotation.from_quat(report['predicted_port_quat_xyzw']) *
                        self.sc_shadow_verifier.nominal_tcp_in_port)
            displacement = np.linalg.norm(position-origin[:3])
            angle = (origin_rot.inv()*rotation).magnitude()
            if displacement > .08 or angle > np.deg2rad(10.):
                self._trace(event='visual_docking_rejected',sim_time=sim,
                            reason='bounded_docking_envelope',displacement_mm=displacement*1000,
                            orientation_change_deg=float(np.rad2deg(angle)))
                return False
            coherent = previous is not None and (
                np.linalg.norm(position-previous[0]) <= .001 and
                (previous[1].inv()*rotation).magnitude() <= np.deg2rad(2.))
            agreeing = agreeing+1 if coherent else 1
            previous = (position,rotation)
            if agreeing >= 2:
                current = Rotation.from_quat(fresh[3:7])
                relative = reference_rotation.inv()*rotation
                rv = relative.as_rotvec()
                magnitude = np.linalg.norm(rv)
                reference_rotation = reference_rotation*Rotation.from_rotvec(rv*min(1.,.035/max(magnitude,1e-9)))
                target_rotation = reference_rotation
                # Resolve posture first; rotation about TCP can move the plug.
                # Advance a persistent reference; do not reset the lookahead
                # from a load-biased measured pose at every camera decision.
                if (current.inv()*rotation).magnitude() <= np.deg2rad(1.):
                    delta = position-reference_position
                    reference_position += delta*min(1.,.005/max(np.linalg.norm(delta),1e-9))
                target_position = reference_position.copy()
                if hasattr(self._parent_node,'_target_mode'):
                    self._parent_node._target_mode=TargetMode.MODE_CARTESIAN
                self.set_pose_target(move_robot=move_robot,
                    pose=build_pose_from_vectors(target_position,target_rotation.as_quat()),
                    frame_id='base_link',stiffness=[240,240,240,50,50,50],
                    damping=[50,50,50,20,20,20])
                self._trace(event='visual_docking_command',sim_time=sim,
                            target_xyz=target_position.tolist(),target_xyzw=target_rotation.as_quat().tolist(),
                            geometry_input=False,supervisor_action=True)
            if report['valid']:
                valid_start = sim if valid_start is None else valid_start
                if sim-valid_start >= 2. and speed <= 5. and angular <= 3.:
                    self._handoff_terminal_snapshot=(fresh.copy(),sim)
                    self._trace(event='matched_handoff_accepted',sim_time=sim,state=fresh.tolist(),
                                report=report,held_valid_s=sim-valid_start,geometry_input=False,
                                route_correction_enabled=True,insertion_enabled=False,
                                tcp_speed_mm_s=speed,angular_speed_deg_s=angular)
                    return True
            else:
                valid_start=None
            if sim-decision_sim >= 15.:
                break
        self._trace(event='visual_docking_rejected',sim_time=last_sim,reason='bounded_docking_timeout')
        return False
