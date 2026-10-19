"""Gated SC RGB grasp re-estimation at the held near-port viewpoint.

Experimental until the exact checkpoint passes episode-held-out pose QA. No
simulator geometry enters inference. SFP retains its existing controller.
"""
import os
import time
import json
from pathlib import Path
import numpy as np
import torch
from .RunVisualConnectorServoRecoveryV13 import RunVisualConnectorServoRecoveryV13
from .RunRuleTransport import RunRuleTransport
from shared_port_tcp_geometry import matrix_from_xyz_xyzw
from close_grasp_consensus import CloseGraspConsensus, compare_grasp_update
from sc_spatial_grasp_observation import SCSpatialGraspObservation
from sc_qualified_close_orientation_observation import SCQualifiedCloseOrientationObservation
from sc_qualified_close_composite_observation import SCQualifiedCloseCompositeObservation
from insertion_reference_governor_v2 import InsertionReferenceGovernor, ReferenceLimits
from scipy.spatial.transform import Rotation


class RunVisualConnectorServoRecoveryV21(RunVisualConnectorServoRecoveryV13):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        approval = json.loads(Path(os.environ['AIC_SC_CLOSE_GRASP_APPROVAL']).read_text())
        experimental = approval.get('prediction_kind') == 'experimental_composite_pose'
        if experimental and os.environ.get('AIC_SC_CLOSE_DIAGNOSTIC_TRIAL') != '1':
            raise ValueError('Unqualified candidate requires an explicit bounded diagnostic run')
        if approval.get('prediction_kind') in ('composite_pose', 'experimental_composite_pose'):
            self.close_grasp_model = SCQualifiedCloseCompositeObservation(
                os.environ['AIC_SC_CLOSE_GRASP_CHECKPOINT'], str(self.local.device),
                approval=os.environ['AIC_SC_CLOSE_GRASP_APPROVAL'], experimental_diagnostic=experimental)
        elif approval.get('prediction_kind') == 'orientation_only':
            self.close_grasp_model = SCQualifiedCloseOrientationObservation(
                os.environ['AIC_SC_CLOSE_GRASP_CHECKPOINT'], str(self.local.device),
                approval=os.environ['AIC_SC_CLOSE_GRASP_APPROVAL'],
                retained_translation_checkpoint=os.environ['AIC_LOCKED_GRASP_CHECKPOINT'])
        else:
            self.close_grasp_model = SCSpatialGraspObservation(
                os.environ['AIC_SC_CLOSE_GRASP_CHECKPOINT'], str(self.local.device),
                approval=os.environ['AIC_SC_CLOSE_GRASP_APPROVAL'])
        if not self.close_grasp_model.approved and not experimental:
            raise ValueError('An independently qualified exact checkpoint is mandatory')
        self._trace(event='close_pose_scope', experimental_trial_only=experimental,
                    model_globally_qualified=self.close_grasp_model.approved, geometry_input=False)
        self.servo_goal_translation = np.zeros(3)
        self.close_update_required = True
        self.close_networks_warmed = False

    @torch.inference_mode()
    def _warm_grasp(self):
        super()._warm_grasp()
        if self.close_networks_warmed:
            return
        # Only initialize network kernels. Synthetic pixels never set a pose,
        # pass an observation gate, or become a controller reference.
        device = self.local.device
        model = self.close_grasp_model
        for _ in range(3):
            if isinstance(model, SCQualifiedCloseCompositeObservation):
                model.position.model(torch.zeros(3, 3, 192, 192, device=device),
                                     camera_id=torch.eye(3, device=device))
                model.rotation.model(torch.zeros(1, 3, 3, 192, 192, device=device,
                                                 dtype=torch.uint8))
            elif getattr(model, 'rotation_only', False):
                model.model(torch.zeros(1, 3, 3, 192, 192, device=device,
                                        dtype=torch.uint8))
            else:
                model.model(torch.zeros(3, 3, 192, 192, device=device),
                            camera_id=torch.eye(3, device=device))
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        self.close_networks_warmed = True
        self._trace(event='close_grasp_networks_prewarmed',
                    synthetic_pixels_used_for_pose=False, geometry_input=False)

    def _lock(self, task_vector, get_observation, move_robot, *, fresh_context):
        accepted = super()._lock(task_vector, get_observation, move_robot, fresh_context=fresh_context)
        if accepted:
            self.close_update_required = True
        return accepted

    def _execute(self, task, get_observation, move_robot, send_feedback, **kwargs):
        self.servo_goal_translation = np.zeros(3)
        return super()._execute(task, get_observation, move_robot, send_feedback, **kwargs)

    def _lateral_retry(self, get_observation, move_robot, *, route_level):
        accepted = super()._lateral_retry(get_observation, move_robot, route_level=route_level)
        if accepted:
            # The legacy parent adds symmetric2.5degree search angles. That
            # would confound testing a newly accurate grasp estimate. Retain
            # its actual backtrack/lateral motion, but target aligned insertion.
            self.servo_goal_rotation = np.eye(3)
            self.servo_goal_translation = np.zeros(3)
            self._trace(event='close_pose_retry_alignment_retained',
                        legacy_angular_search_disabled=True, geometry_input=False)
        return accepted

    def _reestimate_close_grasp(self, get_observation, move_robot):
        # Hold the actual API reference: holding the measured pose would discard
        # the impedance compensation and itself move the physical robot.
        reference, reference_stamp = self.local.governor_seed
        reference = reference.copy()
        old_position = np.asarray(self.local.visual_body_tip).copy()
        old_rotation = np.asarray(self.local.visual_body_connector_rotation).copy()
        consensus = CloseGraspConsensus()
        first_candidate = None
        started = time.monotonic()
        last_sim = None
        while time.monotonic()-started < 10 and time.monotonic() < self.episode_deadline-2:
            observation = get_observation()
            if observation is None:
                time.sleep(.002)
                continue
            stamp = self.local._control_sim_time(observation)
            if last_sim is not None and stamp-last_sim < .099:
                time.sleep(.002)
                continue
            last_sim = stamp
            control_started = time.perf_counter()
            state = self.local.assembler.assemble_ros(observation)
            self._command(move_robot, reference, owner='close_grasp_observation_hold')
            if self._force(state) >= 8 or np.linalg.norm(state[26:29]) > 40:
                self._trace(event='close_grasp_update_rejected', reason='loaded_observation_hold', geometry_input=False)
                return False
            point = np.linalg.inv(self.local.visual_port) @ matrix_from_xyz_xyzw(state[:7])
            tip = point[:3, 3] + point[:3, :3] @ old_position
            if not -.030 <= tip[2] <= -.020:
                self._trace(event='close_grasp_update_rejected', reason='outside_validated_near_view',
                            estimated_axial_m=float(tip[2]), geometry_input=False)
                return False
            if np.linalg.norm(state[7:10]) > .002 or np.linalg.norm(state[13:16]) > .03:
                consensus = CloseGraspConsensus()
                first_candidate = None
                continue
            stamps = [getattr(observation, camera+'_image').header.stamp for camera in ('center', 'left', 'right')]
            stamps = [s.sec*1_000_000_000+s.nanosec for s in stamps]
            # Every image must belong to this causal, synchronized observation.
            ages = [stamp-s*1e-9 for s in stamps]
            if min(ages) < -.001 or max(ages) > .10 or max(stamps)-min(stamps) > 50_000_000:
                self._trace(event='close_grasp_frame_rejected', reason='stale_or_unsynchronized_RGB',
                            ages_s=ages, geometry_input=False)
                continue
            images = {c: RunRuleTransport.native(getattr(observation, c+'_image')) for c in ('center', 'left', 'right')}
            if getattr(self.close_grasp_model, 'rotation_only', False):
                prediction = self.close_grasp_model.predict(images, visual_tip_position_tcp_m=old_position)
            else:
                prediction = self.close_grasp_model.predict(images)
            candidate = consensus.add(stamps, prediction['predicted_tip_position_tcp_m'],
                                      prediction['orientation_for_diagnostic_only'])
            self._trace(event='close_grasp_observation', sim_time=stamp, image_stamps_ns=stamps,
                        consensus=candidate, prediction=prediction,
                        complete_close_estimate_step_ms=(time.perf_counter()-control_started)*1000,
                        geometry_input=False)
            if not candidate['accepted']:
                continue
            change = compare_grasp_update(old_position, old_rotation, candidate)
            if not change['accept_update']:
                if first_candidate is None:
                    first_candidate = candidate
                    consensus = CloseGraspConsensus()
                    # The second window must contain new images as well.
                    consensus.last_stamps = tuple(stamps)
                    self._trace(event='close_grasp_large_disagreement_second_window', **change, geometry_input=False)
                    continue
                self._trace(event='close_grasp_update_rejected', reason='persistent_large_disagreement_reacquisition_required',
                            first_candidate=first_candidate, candidate=candidate, **change, geometry_input=False)
                # Safe abort, not unvalidated interpolation or forward motion.
                return False
            self.local.visual_body_tip = np.asarray(candidate['position_m'])
            self.local.visual_body_connector_rotation = np.asarray(candidate['rotation'])
            self.locked_connector_rotation = self.local.visual_body_connector_rotation.copy()
            self.local.clear_pending_actions()
            self.local.governor = None
            self.local.governor_seed = (reference, stamp)
            self.servo_goal_rotation = np.eye(3)
            self.servo_goal_translation = np.zeros(3)
            self.servo_stall_history = []
            self._trace(event='close_grasp_update_accepted', sim_time=stamp, **change,
                        old_position_tcp_m=old_position.tolist(), old_rotation_tcp=old_rotation.tolist(),
                        candidate=candidate, unchanged_api_reference=reference.tolist(),
                        port_pose_unchanged=True, locked_within_attempt=True,
                        old_estimator_calibration_reused=False, geometry_input=False)
            return True
        self._trace(event='close_grasp_update_rejected', reason='consistent_fresh_observations_unavailable', geometry_input=False)
        return False

    def _visual_servo_insert(self, task, get_observation, move_robot, send_feedback, **kwargs):
        if self.local.task_vector(task)[0] != 1 and self.close_update_required:
            if not self._reestimate_close_grasp(get_observation, move_robot):
                return False
            self.close_update_required = False
        return self._close_pose_servo(task, get_observation, move_robot, send_feedback, **kwargs)

    def _close_pose_servo(self,task,get_observation,move_robot,send_feedback,**kwargs):
        vector=self.local.task_vector(task);port=self.local.visual_port.copy();inverse=np.linalg.inv(port)
        grasp=np.eye(4);grasp[:3,3]=self.local.visual_body_tip;grasp[:3,:3]=self.local.visual_body_connector_rotation
        inverse_grasp=np.linalg.inv(grasp);depth=.0465 if vector[0]==1 else .0156
        depth+=float(self.servo_goal_translation[2])
        # Command margin and estimated-seat acceptance are separate. V18
        # incorrectly raised both; accurate SC cannot physically reach15.8mm.
        command_depth=depth+(0.0004 if vector[0]!=1 else 0.)
        seed,seed_stamp=self.local.governor_seed
        started=time.monotonic();first=None;last=None;stable=None;best=-float('inf');last_progress=None;reference=None;integral=None;governor=None;reason='visual_servo_wall_budget'
        while time.monotonic()-started<60 and time.monotonic()<self.episode_deadline-2:
            observation=get_observation()
            if observation is None:time.sleep(.002);continue
            stamp=self.local._control_sim_time(observation)
            if last is not None and stamp-last<.049:time.sleep(.002);continue
            control_started=time.perf_counter()
            dt=.05 if last is None else min(.1,stamp-last);last=stamp
            state=self.local.assembler.assemble_ros(observation);measured=matrix_from_xyz_xyzw(state[:7]);point=inverse@measured@grasp
            angle=Rotation.from_matrix(self.servo_goal_rotation.T@point[:3,:3]).magnitude();force=self._force(state)
            if first is None:
                first=stamp;last_progress=stamp;reference=np.array([self.servo_goal_translation[0],self.servo_goal_translation[1],point[2,3]])
                integral=(inverse@seed@grasp)[:3,3]-reference
                governor=InsertionReferenceGovernor(port,grasp,seed,measured,seed_stamp,
                    ReferenceLimits(forward_lead_m=.020,alignment_lateral_m=1.,alignment_angle_rad=np.pi,contact_load_n=float('inf')))
                self._trace(event='visual_servo_started',sim_time=stamp,geometry_input=False,actor_owned=False,
                    learned_actor_inference=False,required_seating_dwell_s=2.0,nominal_seating_depth_m=depth,command_depth_m=command_depth,source='frozen RGB port/grasp and public robot state only')
            if point[2,3]>best+.0005:best=float(point[2,3]);last_progress=stamp
            if np.linalg.norm(state[26:29])>40 or (force>=8 and self.mode=='none'):
                reason='visual_servo_force_guard';break
            if stamp-first>35:reason='visual_servo_simulation_budget';break
            if stamp-last_progress>8:reason='visual_servo_no_measured_progress';break
            lateral=float(np.linalg.norm(point[:2,3]-self.servo_goal_translation[:2]));aligned=bool(lateral<=.0005 and angle<=np.deg2rad(.5))
            if aligned:reference[2]=min(command_depth,reference[2]+.006*dt)
            integral=np.clip(integral+.6*dt*(reference-point[:3,3]),-.015,.015)
            desired_point=np.eye(4);desired_point[:3,3]=reference+integral;desired_point[:3,:3]=self.servo_goal_rotation
            desired=port@desired_point@inverse_grasp
            issued,diagnostic=governor.step(desired,measured,stamp,force)
            if self.mode!='none':
                self._supervise(state,issued)
                self._supervise_servo_stall(state,issued,point,stamp,force,depth)
            self._command(move_robot,issued,owner='image_only_insertion_servo_diagnostic')
            self._trace(event='visual_servo_step',sim_time=stamp,estimated_tip_port_m=point[:3,3].tolist(),
                estimated_orientation_error_deg=float(np.degrees(angle)),reference_tip_port_m=reference.tolist(),
                integral_port_m=integral.tolist(),load_change_n=force,aligned=aligned,
                persistent_goal_translation_m=self.servo_goal_translation.tolist(),goal_orientation_offset_deg=np.rad2deg(Rotation.from_matrix(self.servo_goal_rotation).as_rotvec()).tolist(),
                complete_control_step_ms=(time.perf_counter()-control_started)*1000,
                geometry_input=False,actor_owned=False,learned_actor_inference=False,governor=diagnostic)
            seated=(point[2,3]>=depth-.0002 and lateral<=.0005 and angle<=np.deg2rad(.75) and np.linalg.norm(state[7:10])<=.002)
            stable=(stamp if stable is None else stable) if seated else None
            if stable is not None and stamp-stable>=2.0:
                reason='visual_servo_public_estimated_seating';self._trace(event='visual_servo_finished',reason=reason,sim_time=stamp,geometry_input=False,physical_insertion_asserted=False);return True
        self._trace(event='visual_servo_finished',reason=reason,sim_time=last,geometry_input=False,physical_insertion_asserted=False)
        return False
