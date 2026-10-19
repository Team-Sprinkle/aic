"""Experimental same-grasp context fusion; retain the near-only position.

Gazebo fixes the held plug to TCP. This experiment does not claim real slip
handling. Never average the old biased model with the stronger close model.
"""
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .RunVisualConnectorServoRecoveryV26 import RunVisualConnectorServoRecoveryV26
from .RunRuleTransport import RunRuleTransport
from shared_port_tcp_geometry import matrix_from_xyz_xyzw
from close_grasp_consensus import CloseGraspConsensus, compare_grasp_update


class RunVisualConnectorServoContextV28(RunVisualConnectorServoRecoveryV26):
    def _lock(self, task_vector, get_observation, move_robot, *, fresh_context):
        self.far_grasp_frames=[]
        self.far_orientation=None
        return super()._lock(task_vector,get_observation,move_robot,fresh_context=fresh_context)

    def _fresh_report(self, observation, task_vector):
        report,images,state,stamp=super()._fresh_report(observation,task_vector)
        if task_vector[1]==1 and report['valid'] and np.linalg.norm(state[7:10])<=.005 and np.linalg.norm(state[10:13])<=.03:
            stamps=[getattr(observation,c+'_image').header.stamp for c in ('center','left','right')]
            stamps=tuple(x.sec*1000000000+x.nanosec for x in stamps)
            if not self.far_grasp_frames or all(a>b for a,b in zip(stamps,self.far_grasp_frames[-1][0])):
                self.far_grasp_frames=(self.far_grasp_frames+[(stamps,{c:x.copy() for c,x in images.items()})])[-3:]
        return report,images,state,stamp

    def _prepare_far_orientation(self):
        started=time.perf_counter()
        self.far_orientation=None
        if len(self.far_grasp_frames)!=3:
            self._trace(event='far_orientation_context_unavailable',fallback='near_only',geometry_input=False)
            return (time.perf_counter()-started)*1000
        model=getattr(self.close_grasp_model,'rotation',None)
        if model is None:raise ValueError('This context diagnostic requires the explicit composite RGB model')
        rotations=[]
        for stamps,images in self.far_grasp_frames:
            result=model.predict(images,visual_tip_position_tcp_m=self.local.visual_body_tip)
            rotations.append(np.asarray(result['orientation_for_diagnostic_only']))
        span=max(float(np.degrees(Rotation.from_matrix(a.T@b).magnitude())) for a in rotations for b in rotations)
        if span<=.5:self.far_orientation=Rotation.from_matrix(rotations).mean().as_matrix()
        self._trace(event='far_grasp_history_encoded_at25mm',image_stamps_ns=[x[0] for x in self.far_grasp_frames],
            history_not_current_observation=True,rotation_span_deg=span,accepted=span<=.5,
            same_stronger_RGB_model=True,old_RGB_orientation_not_fused=True,
            physical_grasp_rigidity_assumed=True,geometry_input=False,
            elapsed_ms=(time.perf_counter()-started)*1000)
        return (time.perf_counter()-started)*1000

    def _reestimate_close_grasp(self, get_observation, move_robot):
        # Hold the actual API reference: holding the measured pose would discard
        # the impedance compensation and itself move the physical robot.
        far_ms = self._prepare_far_orientation()
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
            control_started = time.perf_counter()-far_ms/1000
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
            far_ms = 0.
            if not candidate['accepted']:
                continue
            if self.far_orientation is not None:
                near = np.asarray(candidate['rotation'])
                difference = float(np.degrees(Rotation.from_matrix(self.far_orientation.T@near).magnitude()))
                if difference <= 1.:
                    candidate['rotation'] = Rotation.from_matrix([self.far_orientation, near]).mean().as_matrix().tolist()
                    candidate['orientation_context_fusion'] = 'equal_stronger_model_far_and_near'
                    candidate['far_near_difference_deg'] = difference
                    candidate['translation_from_near_only'] = True
                else:
                    self._trace(event='far_near_orientation_fusion_rejected',difference_deg=difference,
                                fallback='near_only',geometry_input=False)
            change = compare_grasp_update(old_position, old_rotation, candidate)
            if not change['accept_update']:
                if first_candidate is None:
                    first_candidate = candidate
                    consensus = CloseGraspConsensus()
                    # The second window must contain new images as well.
                    consensus.last_stamps = tuple(stamps)
                    self._trace(event='close_grasp_large_disagreement_second_window', **change, geometry_input=False)
                    continue
                # A second independent window can resolve a small disagreement
                # with the far estimate. Keep the correction bounded; a stable
                # prediction alone is not proof of absolute accuracy.
                repeat_mm=float(np.linalg.norm(np.asarray(first_candidate['position_m'])-
                                               np.asarray(candidate['position_m']))*1000)
                repeat_deg=float(np.degrees(Rotation.from_matrix(
                    np.asarray(first_candidate['rotation']).T@
                    np.asarray(candidate['rotation'])).magnitude()))
                confirmed=bool(change['translation_change_mm']<=1. and
                    change['rotation_change_deg']<=4. and repeat_mm<=.2 and repeat_deg<=.5)
                self._trace(event='close_grasp_two_window_confirmation',
                            accepted=confirmed,repeat_translation_mm=repeat_mm,
                            repeat_rotation_deg=repeat_deg,old_to_near_rotation_deg=change['rotation_change_deg'],
                            geometry_input=False)
                if not confirmed:
                    self._trace(event='close_grasp_update_rejected', reason='persistent_large_disagreement_reacquisition_required',
                                first_candidate=first_candidate, candidate=candidate, **change, geometry_input=False)
                    return False
                change=dict(change,accept_update=True,two_window_confirmation=True)
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
