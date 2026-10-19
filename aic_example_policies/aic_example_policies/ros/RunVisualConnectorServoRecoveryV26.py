"""V21 SC control; bounded SFP seating-force debounce, true angular-speed gate."""
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .RunVisualConnectorServoRecoveryV21 import RunVisualConnectorServoRecoveryV21
from .RunRuleTransport import RunRuleTransport
from shared_port_tcp_geometry import matrix_from_xyz_xyzw
from close_grasp_consensus import CloseGraspConsensus,compare_grasp_update
from insertion_reference_governor_v2 import InsertionReferenceGovernor,ReferenceLimits


class RunVisualConnectorServoRecoveryV26(RunVisualConnectorServoRecoveryV21):
    def _close_pose_servo(self,task,get_observation,move_robot,send_feedback,**kwargs):
        vector=self.local.task_vector(task)
        if vector[0]!=1:
            return super()._close_pose_servo(task,get_observation,move_robot,send_feedback,**kwargs)
        port=self.local.visual_port.copy();inverse=np.linalg.inv(port)
        grasp=np.eye(4);grasp[:3,3]=self.local.visual_body_tip;grasp[:3,:3]=self.local.visual_body_connector_rotation
        inverse_grasp=np.linalg.inv(grasp);depth=.0465 if vector[0]==1 else .0156
        depth+=float(self.servo_goal_translation[2])
        # Command margin and estimated-seat acceptance are separate. V18
        # incorrectly raised both; accurate SC cannot physically reach15.8mm.
        command_depth=depth+(0.0004 if vector[0]!=1 else 0.)
        seed,seed_stamp=self.local.governor_seed
        started=time.monotonic();first=None;last=None;stable=None;best=-float('inf');last_progress=None;reference=None;integral=None;governor=None;reason='visual_servo_wall_budget'
        soft_force_since=None;last_issued=np.asarray(seed).copy()
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
                    learned_actor_inference=False,required_seating_dwell_s=getattr(self,'seating_dwell_s',2.),nominal_seating_depth_m=depth,command_depth_m=command_depth,source='frozen RGB port/grasp and public robot state only')
            if point[2,3]>best+.0005:best=float(point[2,3]);last_progress=stamp
            if np.linalg.norm(state[26:29])>40:
                reason='visual_servo_raw_force_guard';break
            if force>=8 and self.mode=='none':
                near_seat=(point[2,3]>=depth-.0005 and np.linalg.norm(point[:2,3])<=.0005
                           and angle<=np.deg2rad(.75) and np.linalg.norm(state[7:10])<=.005)
                if not near_seat:
                    reason='visual_servo_force_guard_outside_seat';break
                if soft_force_since is None:soft_force_since=stamp
                # Immediately stop demand/progress, with no integration or
                # positive axial command while the threshold is exceeded.
                self._command(move_robot,last_issued,owner='SFP_seating_force_pause')
                self._trace(event='SFP_seating_force_pause',sim_time=stamp,
                    load_change_n=force,raw_force_n=float(np.linalg.norm(state[26:29])),
                    elapsed_s=stamp-soft_force_since,forward_motion_requested=False,
                    held_api_reference=last_issued.tolist(),geometry_input=False)
                if stamp-soft_force_since>=.099:
                    reason='visual_servo_persistent_seat_load_guard';break
                continue
            if soft_force_since is not None:
                self._trace(event='SFP_seating_force_spike_cleared',sim_time=stamp,
                    load_change_n=force,paused_s=stamp-soft_force_since,geometry_input=False)
                soft_force_since=None
            if stamp-first>35:reason='visual_servo_simulation_budget';break
            if stamp-last_progress>8:reason='visual_servo_no_measured_progress';break
            lateral=float(np.linalg.norm(point[:2,3]-self.servo_goal_translation[:2]));aligned=bool(lateral<=.0005 and angle<=np.deg2rad(.5))
            if aligned:reference[2]=min(command_depth,reference[2]+getattr(self,'insertion_axial_speed_m_s',.006)*dt)
            integral=np.clip(integral+.6*dt*(reference-point[:3,3]),-.015,.015)
            desired_point=np.eye(4);desired_point[:3,3]=reference+integral;desired_point[:3,:3]=self.servo_goal_rotation
            desired=port@desired_point@inverse_grasp
            issued,diagnostic=governor.step(desired,measured,stamp,force)
            if self.mode!='none':
                self._supervise(state,issued)
                self._supervise_servo_stall(state,issued,point,stamp,force,depth)
            self._command(move_robot,issued,owner='image_only_insertion_servo_diagnostic')
            last_issued=issued.copy()
            self._trace(event='visual_servo_step',sim_time=stamp,estimated_tip_port_m=point[:3,3].tolist(),
                estimated_orientation_error_deg=float(np.degrees(angle)),reference_tip_port_m=reference.tolist(),
                integral_port_m=integral.tolist(),load_change_n=force,aligned=aligned,
                persistent_goal_translation_m=self.servo_goal_translation.tolist(),goal_orientation_offset_deg=np.rad2deg(Rotation.from_matrix(self.servo_goal_rotation).as_rotvec()).tolist(),
                complete_control_step_ms=(time.perf_counter()-control_started)*1000,
                geometry_input=False,actor_owned=False,learned_actor_inference=False,governor=diagnostic)
            seated=(point[2,3]>=depth-.0002 and lateral<=.0005 and angle<=np.deg2rad(.75) and np.linalg.norm(state[7:10])<=.002)
            stable=(stamp if stable is None else stable) if seated else None
            if stable is not None and stamp-stable>=getattr(self,'seating_dwell_s',2.):
                reason='visual_servo_public_estimated_seating';self._trace(event='visual_servo_finished',reason=reason,sim_time=stamp,geometry_input=False,physical_insertion_asserted=False);return True
        self._trace(event='visual_servo_finished',reason=reason,sim_time=last,geometry_input=False,physical_insertion_asserted=False)
        return False

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
            if np.linalg.norm(state[7:10]) > .002 or np.linalg.norm(state[10:13]) > .03:
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
