"""Training-envelope SC corridor with measured completion and visual approval.

No scene configuration or simulator target transform is read by this policy.
This development controller stops at acquisition; insertion stays disabled.
"""
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch

from .RunSharedPortTCP import RunSharedPortTCP, REPO_ROOT
from shared_port_tcp_geometry import matrix_from_xyz_xyzw
from sc_shadow_handoff_verifier import SCShadowHandoffVerifier


class RunRuleTransport(RunSharedPortTCP):
    def __init__(self, parent_node):
        torch.set_num_threads(4)
        cv2.setNumThreads(2)
        super().__init__(parent_node)
        self.target_frame = 'body'
        self.transport_terminal_reason = 'rule_acquisition_or_guarded_failure'
        self.execute_steps = 4
        self.template = json.loads(Path(os.environ['AIC_RULE_TRANSPORT_TEMPLATE']).read_text())
        if self.template['geometry_runtime_input']:
            raise ValueError('Privileged rule template is prohibited')
        root = REPO_ROOT/'outputs/experiments/2026-10-02_two_family_perception'
        self.verifier = SCShadowHandoffVerifier(
            root/'joint_locator_v1/best.pt',
            '/var/tmp/chmin_aic_20260918_act/shadow_handoff_20261003/sc_spatial_identity.pt',
            '/var/tmp/chmin_aic_20260918_act/shadow_handoff_20261003/sc_orientation_oneport.pt',
            os.environ.get('AIC_JOINT_NATIVE_REFINER_CHECKPOINT') or str(root/'sc_native_refiner_v1/best.pt'),
            REPO_ROOT/'configs/hierarchical_recovery/sc_visual_standoff_20261001.json',
            REPO_ROOT/'configs/hierarchical_recovery/sc_board_to_port_training_fixed.json',
            root/'local_sc_orientation_prior_v1.json', device=str(self.device),
            candidate_association_path=os.environ.get('AIC_SC_CANDIDATE_ASSOCIATION_CHECKPOINT') or None,
            candidate_gate_mode=os.environ.get('AIC_SC_CANDIDATE_GATE_MODE') or 'intersection',
            requested_locator_path=os.environ.get('AIC_SC_REQUESTED_HISTORY_LOCATOR_CHECKPOINT') or None)
        self.rule_stages = None
        self.rule_index = 0
        self.completion_streak = 0
        self.rule_accept_streak = 0
        self.rule_final_valid = False
        self.rule_best_error = float('inf')
        self.rule_last_progress_s = None
        self.rule_failure = None
        self.rule_visual_goal = None
        self.rule_reference = None
        self.rule_hold_start_s = None
        self.maximum_visual_reacquisition_m=.08

    @staticmethod
    def native(message):
        channels = 4 if message.encoding.lower() in ('rgba8', 'bgra8') else 3
        raw = np.frombuffer(message.data, np.uint8).reshape(message.height, message.step)
        rgb = raw[:, :message.width*channels].reshape(message.height, message.width, channels)[..., :3]
        return rgb[..., ::-1].copy() if message.encoding.lower().startswith('bgr') else rgb.copy()

    def predict_chunk(self, observation, task_vector, *, decision_time_s=None):
        tick = time.perf_counter()
        if task_vector[0] == 1:
            raise ValueError('SC-only rule feasibility controller; SFP is a separate branch')
        state = self.assembler.assemble_ros(observation)
        sim = self._control_sim_time(observation)
        if self.first_sim_time is None:
            self.first_sim_time = sim
        self.last_elapsed_sim_sec = sim-self.first_sim_time
        current = matrix_from_xyz_xyzw(state[:7])
        if self.verifier.candidate_association is not None or self.verifier.requested_locator is not None:
            images={c:self.native(getattr(observation,c+'_image')) for c in ('center','left','right')}
            self.verifier.retain_rgb_context(images,state)
        if self.rule_reference is None:
            self.rule_reference = current.copy()
        requested = int(np.argmax(task_vector[2:4]))
        t = self.template
        if self.rule_stages is None:
            initial = current[:3, :3]
            aligned = Rotation.from_quat(t['nominal_tcp_xyzw']).as_matrix()
            final = t['final_by_requested_port'][str(requested)]
            x, behind, beyond, low = t['outside_x_m'], t['behind_y_m'], t['beyond_y_m'], t['low_z_m']
            self.rule_stages = [
                ('clear_start_side', [float(state[0]), behind, max(low+.05, float(state[2])-.04)], initial),
                ('shift_behind_row_low', [x, behind, low], initial),
                ('traverse_beyond_row', [x, beyond, low], initial),
                ('sweep_after_row', [final[0], beyond, low], initial),
                ('tilt_clear_of_cards', [final[0], beyond, low], aligned),
                ('return_toward_acquisition', final, aligned)]
            self.rule_last_progress_s = sim
        stage, xyz, rotation = self.rule_stages[self.rule_index]
        goal = np.eye(4)
        goal[:3, 3], goal[:3, :3] = xyz, rotation
        if self.rule_visual_goal is not None:
            goal = self.rule_visual_goal.copy()
        error = float(np.linalg.norm(goal[:3, 3]-state[:3]))
        angle = float((Rotation.from_matrix(goal[:3,:3]).inv()*Rotation.from_quat(state[3:7])).magnitude())
        progress_error = error+.10*angle
        if progress_error < self.rule_best_error-.002:
            self.rule_best_error = progress_error
            self.rule_last_progress_s = sim
        complete = error <= t['phase_completion_position_m'] and angle <= np.deg2rad(t['phase_completion_orientation_deg'])
        self.completion_streak = self.completion_streak+1 if complete else 0
        if self.completion_streak >= 3 and self.rule_index < len(self.rule_stages)-1:
            self._trace(event='rule_stage_completed', stage=stage, sim_time=sim,
                        position_error_mm=error*1000, angle_error_deg=float(np.rad2deg(angle)))
            self.rule_index += 1
            self.completion_streak = 0
            self.rule_best_error = float('inf')
            self.rule_last_progress_s = sim
            self.rule_reference = current.copy()
            stage, xyz, rotation = self.rule_stages[self.rule_index]
            goal[:3, 3], goal[:3, :3] = xyz, rotation
        if sim-self.rule_last_progress_s > 10:
            self.rule_failure = 'stage_no_measured_progress'
        if self.rule_index == len(self.rule_stages)-1 and not self.rule_failure:
            images = {c:self.native(getattr(observation, c+'_image')) for c in ('center','left','right')}
            report = self.verifier.inspect(images, state, requested)
            self._trace(event='rule_visual_inspection', sim_time=sim, requested_port=requested,
                        report=report, geometry_input=False)
            self.rule_accept_streak = self.rule_accept_streak+1 if report['valid'] else 0
            speed=float(np.linalg.norm(state[7:10]))
            angular_speed=float(np.linalg.norm(state[10:13]))
            if self.rule_accept_streak >= 3:
                if self.rule_hold_start_s is None:
                    self.rule_hold_start_s=sim
            else:
                self.rule_hold_start_s=None
            self.rule_final_valid=bool(self.rule_hold_start_s is not None and
                                       sim-self.rule_hold_start_s>=2. and
                                       speed<=.005 and angular_speed<=np.deg2rad(3.))
            self._trace(event='rule_hold_gate',sim_time=sim,
                        valid_visual_streak=self.rule_accept_streak,
                        held_s=(sim-self.rule_hold_start_s if self.rule_hold_start_s is not None else 0.),
                        tcp_speed_mm_s=speed*1000,angular_speed_deg_s=float(np.rad2deg(angular_speed)),
                        passed=self.rule_final_valid)
            # Only a coherent, correctly requested visual pose may adjust the
            # last acquisition waypoint. Bound reacquisition to the local area.
            if report['reason'] in ('not_at_selected_standoff', 'wrong_tcp_posture'):
                predicted = np.asarray(report['predicted_opening_base_m'])+self.verifier.standoff_offset
                if np.linalg.norm(predicted-np.asarray(xyz)) <= self.maximum_visual_reacquisition_m:
                    goal[:3, 3] = predicted
                    goal[:3, :3] = (Rotation.from_quat(report['predicted_port_quat_xyzw'])*self.verifier.nominal_tcp_in_port).as_matrix()
                    self.rule_visual_goal = goal.copy()
        if self.rule_failure:
            goal = current.copy()
            self.rule_reference = current.copy()
        # Advance the commanded reference, rather than continually resetting
        # its lookahead from a load-biased measured pose. Stage completion is
        # still measured, and the shared executor bounds tracking error.
        reference = self.rule_reference
        distance = float(np.linalg.norm(goal[:3, 3]-reference[:3, 3]))
        rotation_distance = float((Rotation.from_matrix(reference[:3,:3]).inv()*Rotation.from_matrix(goal[:3,:3])).magnitude())
        duration = max(distance/.08, rotation_distance/.35, .2)
        interpolator = Slerp([0., 1.], Rotation.from_matrix(np.stack([reference[:3,:3], goal[:3,:3]])))
        targets = []
        for j in range(4):
            alpha = min((j+1)/self.control_hz/duration, 1.)
            target = np.eye(4)
            target[:3,3] = reference[:3,3]+alpha*(goal[:3,3]-reference[:3,3])
            target[:3,:3] = interpolator([alpha]).as_matrix()[0]
            relative = np.linalg.inv(current)@target
            targets.append(np.r_[relative[:3,3], relative[:3,:2].T.reshape(-1)])
        self.rule_reference = target.copy()
        self._trace(event='rule_progress', stage=stage, index=self.rule_index, sim_time=sim,
                    position_error_mm=error*1000, angle_error_deg=float(np.rad2deg(angle)),
                    failure=self.rule_failure, geometry_input=False)
        self._inference_latency_ms.append((time.perf_counter()-tick)*1000)
        return np.asarray(targets, np.float32), np.zeros(9,np.float32), state

    def _should_stop_transport(self, task_vector):
        return self.rule_final_valid or self.rule_failure is not None

    def reset_for_task(self, task):
        super().reset_for_task(task)
        self.verifier.reset_context()
        self.rule_stages = None
        self.rule_index = 0
        self.completion_streak = 0
        self.rule_accept_streak = 0
        self.rule_final_valid = False
        self.rule_best_error = float('inf')
        self.rule_last_progress_s = None
        self.rule_failure = None
        self.rule_visual_goal = None
        self.rule_reference = None
        self.rule_hold_start_s = None

    def _inspect_transport_handoff(self, observed, state, task_vector, get_observation, move_robot, decision_sim):
        self._trace(event='rule_terminal', sim_time=decision_sim, state=state.tolist(),
                    acquired=self.rule_final_valid, failure=self.rule_failure, insertion_enabled=False)
        return True
