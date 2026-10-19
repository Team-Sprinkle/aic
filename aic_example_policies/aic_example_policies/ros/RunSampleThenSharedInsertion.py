"""Frozen sample-derived transport followed by the trained shared 150mm actor.

Only public task/state/RGB enter decisions. Scoring and contact identities are
recorded externally for posthoc assessment. This is supervised BC plus a rule
supervisor, not RL. Keep the previous pipeline intact for comparison.
"""
import os
import time
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .RunRuleThenSharedRecovery import RunRuleThenSharedRecovery, RecoveryTransfer, measured_pose
from .RunSampleTransportRecovery import RunSampleTransportRecovery
from .RunRuleTransport import RunRuleTransport
from .hierarchical_recovery import HierarchicalRecovery, RecoveryConfig
from sc_native_grasp_observation import SCNativeGraspObservation
from sfp_native_grasp_observation import SFPNativeGraspObservation


class RunSampleThenSharedInsertion(RunRuleThenSharedRecovery):
    def _fresh_report(self, observation, task_vector):
        state = self.local.assembler.assemble_ros(observation)
        sim = self.local._control_sim_time(observation)
        images = {}
        if getattr(self,'parallel_fresh_JPEG',False):
            from parallel_training_jpeg import normalise_triplet
            images=normalise_triplet({camera:RunRuleTransport.native(getattr(observation,camera+'_image'))
                                      for camera in ('center','left','right')})
        else:
            for camera in ('center', 'left', 'right'):
                rgb = RunRuleTransport.native(getattr(observation, camera + '_image'))
                ok, encoded = cv2.imencode('.jpg', rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 86])
                if not ok:
                    raise RuntimeError('Native RGB codec failed')
                images[camera] = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        prediction = self.rule.component.predict(images, state, task_vector)
        report = dict(valid=False, reason=prediction['reason'], runtime_true_geometry=False)
        if prediction['accepted']:
            port = np.asarray(prediction['base_port_matrix'])
            # Use the same broad SFP envelope as transport. The sample-derived
            # nominal can be >0.20 m from a valid released-board placement;
            # the requested three-view fit and fresh temporal lock still gate
            # this pose before any insertion command.
            envelope = .35 if task_vector[0] == 1 else .20
            within = np.linalg.norm(port[:3, 3] - self.rule.nominal_port[:3, 3]) <= envelope
            report.update(valid=bool(within),
                reason='fresh_requested_native_pose' if within else 'outside_public_sample_search_envelope',
                predicted_opening_base_m=port[:3, 3].tolist(),
                predicted_port_quat_xyzw=Rotation.from_matrix(port[:3, :3]).as_quat().tolist())
        return report, images, state, sim

    def _make_rule_transport(self):
        return RunSampleTransportRecovery(self.parent)

    def _execute(self, task, get_observation, move_robot, send_feedback, **kwargs):
        self.exploration_rng = np.random.default_rng(self.exploration_seed)
        task_vector = self.local.task_vector(task)
        self.rule = self._make_rule_transport()
        if task_vector[1] == 1:
            self.grasp = SCNativeGraspObservation(os.environ['AIC_LOCKED_GRASP_CHECKPOINT'], str(self.local.device))
        else:
            self.grasp = SFPNativeGraspObservation(os.environ['AIC_SFP_GRASP_CHECKPOINT'],
                os.environ['AIC_SFP_GRASP_APPROVAL'], str(self.local.device))
        self._warm_grasp()
        self.pipeline_phase = 'initial_force_reference'
        if not self._initial_force_reference(get_observation, move_robot):
            return False
        self.pipeline_phase = 'rule_transport'
        self.rule.insert_cable(task, get_observation, move_robot, send_feedback, **kwargs)
        # The transport helper deliberately returns False because it does not
        # assert insertion. Its explicit acquisition flag is the contract.
        if not self.rule.rule_final_valid or self.rule.rule_failure:
            return False
        context = SimpleNamespace(reset_context=self.rule.component.reset)
        self.rule.verifier = context
        self.rule.observer = context
        observation = get_observation()
        if observation is None:
            return False
        self.acquisition_state = self.local.assembler.assemble_ros(observation)
        self.pipeline_phase = 'fresh_visual_acquisition'
        if not self._lock(task_vector, get_observation, move_robot, fresh_context=True):
            return False
        self.supervisor = HierarchicalRecovery(RecoveryConfig(max_total_backoffs=8, max_history=4096,
            local_attempts_before_route=2 if self.mode == 'hierarchical' else 100))
        self.last_target = None
        self.last_command_m = 0.
        self.route = 'acquisition_0'
        self.supervisor.step(pose=measured_pose(self.acquisition_state),
            force_n=self._force(self.acquisition_state), prior_command_m=0.,
            phase='acquisition_checkpoint', route=self.route)
        current = get_observation()
        if current is None:
            return False
        self.insertion_start_sim = self.local._control_sim_time(current)
        self.local.supervisor_callback = self._supervise if self.mode != 'none' else None
        while True:
            current = get_observation()
            if current is None:
                return False
            remaining = self.total_budget_s - (self.local._control_sim_time(current) - self.insertion_start_sim)
            remaining_wall = self.episode_deadline - time.monotonic()
            if remaining <= 0 or remaining_wall <= 2:
                return False
            self.local.max_simulation_sec = remaining
            self.local.max_runtime_sec = remaining_wall - 1.
            self.pipeline_phase = 'actor'
            try:
                return self.local.insert_cable(task, get_observation, move_robot, send_feedback, **kwargs)
            except RecoveryTransfer as transfer:
                self.last_target = None
                self.last_command_m = 0.
                self.pipeline_phase = 'measured_backtrack'
                if not self._retreat(transfer, get_observation, move_robot):
                    return False
                route_level = transfer.decision.level == 'route'
                if route_level:
                    self.pipeline_phase = 'fresh_reacquisition'
                    if not self._lock(task_vector, get_observation, move_robot, fresh_context=True):
                        return False
                self.pipeline_phase = 'supervisor_lateral_retry'
                if not self._lateral_retry(get_observation, move_robot, route_level=route_level):
                    return False
                if route_level:
                    self.route = self.supervisor.confirm_new_corridor(
                        f'acquisition_{self.supervisor.route_resets}', self.last_measured_lateral_retry_m)
                self.local.clear_pending_actions()
                self.local.robot_history = []
