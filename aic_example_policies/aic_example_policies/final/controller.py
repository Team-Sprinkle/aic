"""Final explicit controller; retained layers preserve selected method dispatch.

Only methods reachable from the selected controller and transports are retained.
Historical private layer names identify provenance, not alternative policies.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .runtime.cartesian_hold_gain_transfer import transfer_hold
from .runtime.shared_port_tcp_geometry import matrix_from_xyz_xyzw
import os
from .runtime.causal_force_reference import InitialLoadReference
import torch
import time
from aic_model.policy import Policy, build_pose_from_vectors
import cv2
from aic_control_interfaces.msg import TargetMode
from .runtime.hierarchical_recovery import MeasuredPose
from pathlib import Path
from .runtime.native_connector_pose_observation import NativeConnectorPoseObservation
from .runtime.close_grasp_consensus import CloseGraspConsensus, compare_grasp_update
from .runtime.requested_native_port_pose import RequestedNativePortPose
from .runtime.parallel_training_jpeg import normalise_triplet
from .runtime.observed_board_marker import estimate_board_marker
from scipy.spatial.transform import Rotation, Slerp
import json
from aic_model.policy import Policy, build_pose_from_vectors, clamp_delta_pose_components
from .runtime.shared_port_tcp_geometry import inferred_base_port, body_delta_from_target, matrix_from_xyz_xyzw, matrix_from_pose9
from types import SimpleNamespace
from .runtime.lerobot_robot_aic_runtime_features import AICRuntimeFeatureAssembler
from aic_task_interfaces.msg import Task
from .runtime.insertion_reference_governor_v2 import InsertionReferenceGovernor, ReferenceLimits
from .runtime.empty_scene_receipt import EmptySceneConsensus
from .runtime.sc_native_grasp_observation import SCNativeGraspObservation
from .runtime.hierarchical_recovery import HierarchicalRecovery, RecoveryConfig
from .runtime.sfp_native_grasp_observation import SFPNativeGraspObservation
from .runtime.empty_scene_receipt import receipt_matches
from .runtime.shared_grasp_training_jpeg import SharedGraspTrainingJPEG, SharedJPEGEstimator
from .runtime.lerobot_robot_aic_runtime_features import base_state_from_ros_observation
from .runtime.stable_pose_window import StablePoseWindow
from aic_model.policy import build_pose_from_vectors
from .runtime.transport_local_recovery import TransportLoadStallMonitor, measured_approach_axis, horizontal_lateral_axis
import copy
from .runtime.empty_row_observation import EmptyRowObservation
from .runtime.geometry_sc_native_triangulation import FOCAL, triangulate
from .runtime.approach_reference import ApproachReference
from .runtime.approach_orientation_reference import ApproachOrientationReference
from .runtime.correlated_lateral_retry import lateral_retry_direction
from .runtime.public_packet_camera_timing import camera_receipt
from .runtime.sc_qualified_close_composite_observation import SCQualifiedCloseCompositeObservation
from .runtime.sc_qualified_close_orientation_observation import SCQualifiedCloseOrientationObservation
from .runtime.sc_spatial_grasp_observation import SCSpatialGraspObservation
from types import SimpleNamespace

def measured_pose(state):
    q = state[3:7]
    return MeasuredPose(tuple(state[:3]), (q[3], q[0], q[1], q[2]))
ROOT = Path(os.environ['AIC_CONNECTOR_MODEL_ROOT'])

class _RunSharedPortTCP(Policy):
    supports_sc_tip_target_reference = False

    @staticmethod
    def task_vector(task):
        if task.target_module_name.startswith('nic_card_mount_'):
            card = int(task.target_module_name.rsplit('_', 1)[-1])
            port = int(task.port_name.rsplit('_', 1)[-1])
            if card not in range(5) or port not in (0, 1):
                raise ValueError('Invalid SFP task encoding')
            one_hot = [0.0] * 5
            one_hot[card] = 1.0
            return [1.0, 0.0, float(port == 0), float(port == 1), *one_hot, 1.0]
        if task.target_module_name.startswith('sc_port_'):
            port = int(task.target_module_name.rsplit('_', 1)[-1])
            if port not in (0, 1):
                raise ValueError('Invalid SC target port')
            return [0.0, 1.0, float(port == 0), float(port == 1), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        raise ValueError('Unknown AIC target module')

    def reset_for_task(self, task):
        """Reset causal state for bounded teacher-collection interventions."""
        self.clear_pending_actions()
        self.assembler.reset()
        self.first_sim_time = None
        self.last_elapsed_sim_sec = 0.0
        self.student_proposal_count = 0
        self._inference_latency_ms.clear()
        self._student_task_vector = self.task_vector(task)

    def clear_pending_actions(self):
        self._action_generation += 1

    def _target_anchor(self, tcp_base, estimated_pose):
        return inferred_base_port(tcp_base, estimated_pose) if self.target_frame in ('port', 'estimated_port') else matrix_from_xyz_xyzw(tcp_base)

    def _body_delta(self, tcp_base, anchor, target):
        if self.target_frame in ('port', 'estimated_port'):
            return body_delta_from_target(tcp_base, anchor, target)
        relative = np.linalg.inv(matrix_from_xyz_xyzw(tcp_base)) @ anchor @ matrix_from_pose9(target)
        return np.concatenate((relative[:3, 3], Rotation.from_matrix(relative[:3, :3]).as_rotvec()))

    def _cartesian_gains(self):
        return (self.cartesian_stiffness, self.cartesian_damping)

    def _bound_transport_target(self, absolute, task_vector, state):
        return absolute

    def insert_cable(self, task: Task, get_observation, move_robot, send_feedback, **kwargs) -> bool:
        self.reset_for_task(task)
        task_vector = self.task_vector(task)
        start = time.monotonic()
        issued = 0
        period = 1 / self.control_hz
        first_control_sim = None
        next_decision_sim = None
        resume_after_rejection = False
        while time.monotonic() - start < self.max_runtime_sec:
            observed = get_observation()
            if observed is None:
                time.sleep(period)
                continue
            decision_sim = self._control_sim_time(observed)
            if first_control_sim is None:
                first_control_sim = decision_sim
            if self.control_clock == 'simulation':
                if decision_sim - first_control_sim >= self.max_simulation_sec:
                    break
                if next_decision_sim is not None and decision_sim < next_decision_sim:
                    time.sleep(0.002)
                    continue
            plan_generation = self._action_generation
            targets, estimated_pose, anchor_state = self.predict_chunk(observed, task_vector, decision_time_s=issued / self.control_hz)
            self._trace(event='proposal', sim_time=decision_sim, state=anchor_state.tolist(), targets=targets.tolist(), task_vector=task_vector, masked_state_indices=self.control_metadata.masked_state_indices, image_sim_ns={name: getattr(observed, name + '_image').header.stamp.sec * 1000000000 + getattr(observed, name + '_image').header.stamp.nanosec for name in ('center', 'left', 'right')}, inference_ms=self._inference_latency_ms[-1], geometry_input=False)
            should_inspect = not resume_after_rejection and self._should_stop_transport(task_vector)
            resume_after_rejection = False
            if should_inspect:
                self._handoff_terminal_snapshot = None
                if not self._inspect_transport_handoff(observed, anchor_state, task_vector, get_observation, move_robot, decision_sim):
                    next_decision_sim = decision_sim + period
                    resume_after_rejection = True
                    if hasattr(self, 'stop_streak'):
                        self.stop_streak = 0
                    continue
                snapshot = getattr(self, '_handoff_terminal_snapshot', None)
                terminal_state, terminal_sim = snapshot[:2] if snapshot is not None else (anchor_state, decision_sim)
                self._trace(event='terminal_before_return', reason=getattr(self, 'transport_terminal_reason', 'learned_transport_handoff'), state=terminal_state.tolist(), sim_time=terminal_sim, observation_source='accepted_handoff_observation' if snapshot is not None else 'current_decision_observation', geometry_input=False)
                self._report_latency()
                return False
            anchor = self._target_anchor(anchor_state[:7], estimated_pose)
            planned = targets[:1] if self.replan_every_command else targets[:self.execute_steps]
            next_scheduled_sim = None
            if self.execute_complete_chunk:
                fresh = get_observation()
                if fresh is None:
                    break
                next_scheduled_sim = max(decision_sim, self._control_sim_time(fresh))
            for index, target in enumerate(planned):
                if plan_generation != self._action_generation:
                    self._trace(event='chunk_cancelled_for_control_transfer', index=index)
                    break
                if time.monotonic() - start >= self.max_runtime_sec:
                    break
                tick = time.monotonic()
                scheduled_sim = next_scheduled_sim if self.execute_complete_chunk else decision_sim + index * period
                if self.control_clock == 'simulation':
                    current = get_observation()
                    while current is not None and self._control_sim_time(current) < scheduled_sim and (time.monotonic() - start < self.max_runtime_sec):
                        time.sleep(0.002)
                        current = get_observation()
                    if current is None:
                        break
                    if not self.execute_complete_chunk and index > 0 and (self._control_sim_time(current) > scheduled_sim + period):
                        self._trace(event='stale_chunk_member_dropped', index=index, sim_time=self._control_sim_time(current), scheduled_sim=scheduled_sim)
                        continue
                    state = self.assembler.assemble_ros(current)
                elif index == 0:
                    state = anchor_state
                else:
                    current = get_observation()
                    if current is None:
                        break
                    state = self.assembler.assemble_ros(current)
                if plan_generation != self._action_generation:
                    self._trace(event='chunk_cancelled_for_control_transfer', index=index)
                    break
                delta = np.zeros(6) if self.last_elapsed_sim_sec < self.startup_hold_sim_sec else self._body_delta(state[:7], anchor, target)
                if not np.isfinite(delta).all():
                    raise ValueError('Nonfinite predicted port-frame TCP command')
                force_norm = float(np.linalg.norm(state[26:29]))
                if force_norm > self.diagnostic_force_stop_n:
                    self.get_logger().error(f'Shared BC diagnostic force stop: {force_norm:.1f} N')
                    retained_observation = current if self.control_clock == 'simulation' or index > 0 else observed
                    self._trace(event='terminal_before_return', reason='force_stop', sim_time=self._control_sim_time(retained_observation), state=state.tolist(), force_norm_n=force_norm, observation_source='current_command_tick', post_reset_geometry_used=False)
                    self._report_latency()
                    return False
                translation, rotation = clamp_delta_pose_components(delta[:3], delta[3:], max_translation=self.max_translation, max_rotation=self.max_rotation)
                body_target = np.eye(4)
                body_target[:3, 3] = translation
                body_target[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
                absolute = matrix_from_xyz_xyzw(state[:7]) @ body_target
                absolute = self._bound_transport_target(absolute, task_vector, state)
                effective_body = np.linalg.inv(matrix_from_xyz_xyzw(state[:7])) @ absolute
                effective_delta = np.r_[effective_body[:3, 3], Rotation.from_matrix(effective_body[:3, :3]).as_rotvec()]
                pose = build_pose_from_vectors(absolute[:3, 3], Rotation.from_matrix(absolute[:3, :3]).as_quat())
                if hasattr(self._parent_node, '_target_mode'):
                    self._parent_node._target_mode = TargetMode.MODE_CARTESIAN
                stiffness, damping = self._cartesian_gains()
                if plan_generation != self._action_generation:
                    self._trace(event='chunk_cancelled_for_control_transfer', index=index)
                    break
                self.set_pose_target(move_robot=move_robot, pose=pose, frame_id='base_link', stiffness=stiffness, damping=damping)
                self._trace(event='command', sim_time=self._control_sim_time(current) if self.control_clock == 'simulation' else decision_sim, scheduled_sim=scheduled_sim, chunk_index=index, state=state.tolist(), delta=effective_delta.tolist(), absolute_tcp_target=absolute.tolist(), stiffness=stiffness, damping=damping)
                if self.execute_complete_chunk:
                    next_scheduled_sim = self._control_sim_time(current) + period
                issued += 1
                if issued % max(1, round(self.control_hz)) == 0:
                    send_feedback(f'in progress; shared port TCP commands={issued}')
                if self.control_clock == 'wall':
                    time.sleep(max(0.0, period - (time.monotonic() - tick)))
            next_decision_sim = next_scheduled_sim if self.execute_complete_chunk else decision_sim + len(planned) * period
        terminal = get_observation()
        if terminal is not None:
            self._trace(event='terminal_before_return', reason='bounded_runtime', sim_time=self._control_sim_time(terminal), state=self.assembler.assemble_ros(terminal).tolist())
        self._report_latency()
        return False

    @staticmethod
    def _control_sim_time(observation):
        stamp = observation.controller_state.header.stamp
        return float(stamp.sec) + float(stamp.nanosec) * 1e-09

    def _trace(self, **record):
        if self.trace_path:
            with open(self.trace_path, 'a') as handle:
                handle.write(json.dumps(record) + '\n')

    def _report_latency(self):
        if self._inference_latency_ms:
            values = np.asarray(self._inference_latency_ms)
            p50, p95, p99 = np.quantile(values, [0.5, 0.95, 0.99])
            self.get_logger().info(f'Shared BC camera/state/model inference latency ms: n={len(values)} p50={p50:.2f} p95={p95:.2f} p99={p99:.2f}')

    def __init__(self, parent_node, checkpoint_path=None):
        Policy.__init__(self, parent_node)
        self.trained_families = ('sc', 'sfp')
        self.recorded_controller_contract = None
        self.device = torch.device(os.environ.get('AIC_SHARED_PORT_TCP_DEVICE', 'cpu'))
        self.include_tip = True
        self.tip_input_mode = 'full_pose'
        self.sc_action_reference = 'full_connector_pose'
        self.target_frame = 'body'
        self.history_offsets_s = [1.2, 0.8, 0.4, 0.0]
        self.history_masked = False
        self.assembler = AICRuntimeFeatureAssembler(32, fps=20.0)
        self.control_metadata = SimpleNamespace(masked_state_indices=[], action_horizon=4, use_local_history=True)
        self.control_hz = float(os.environ.get('AIC_SHARED_PORT_TCP_CONTROL_HZ', '20'))
        self.control_clock = os.environ.get('AIC_SHARED_PORT_TCP_CONTROL_CLOCK', 'wall')
        if self.control_clock not in ('wall', 'simulation'):
            raise ValueError('Unknown shared actor control clock')
        self.max_simulation_sec = float(os.environ.get('AIC_SHARED_PORT_TCP_MAX_SIMULATION_SEC', '70'))
        self.trace_path = os.environ.get('AIC_SHARED_PORT_TCP_TRACE', '')
        self.cartesian_stiffness = json.loads(os.environ.get('AIC_SHARED_PORT_TCP_CARTESIAN_STIFFNESS', '[90,90,90,50,50,50]'))
        self.cartesian_damping = json.loads(os.environ.get('AIC_SHARED_PORT_TCP_CARTESIAN_DAMPING', '[50,50,50,20,20,20]'))
        if any((len(v) != 6 or not all((np.isfinite(x) and x > 0 for x in v)) for v in (self.cartesian_stiffness, self.cartesian_damping))):
            raise ValueError('Invalid Cartesian gain vector')
        self.max_runtime_sec = float(os.environ.get('AIC_SHARED_PORT_TCP_MAX_RUNTIME_SEC', '90'))
        self.max_translation = float(os.environ.get('AIC_SHARED_PORT_TCP_MAX_TRANSLATION_DELTA', '.02'))
        self.max_rotation = float(os.environ.get('AIC_SHARED_PORT_TCP_MAX_ROTATION_DELTA', '.2'))
        self.replan_every_command = os.environ.get('AIC_SHARED_PORT_TCP_REPLAN_EVERY_COMMAND', '0') == '1'
        self.execute_steps = None
        self._action_generation = 0
        self.execute_complete_chunk = os.environ.get('AIC_SHARED_PORT_TCP_EXECUTE_COMPLETE_CHUNK', '0') == '1'
        if self.execute_complete_chunk and (self.control_clock != 'simulation' or self.replan_every_command):
            raise ValueError('Complete-chunk execution requires simulation clock and chunk control')
        self.diagnostic_force_stop_n = float(os.environ.get('AIC_SHARED_PORT_TCP_DIAGNOSTIC_FORCE_STOP_N', '40'))
        self.startup_hold_sim_sec = float(os.environ.get('AIC_SHARED_PORT_TCP_STARTUP_HOLD_SIM_SEC', '0'))
        if self.startup_hold_sim_sec < 0:
            raise ValueError('Startup hold duration must be nonnegative')
        self.first_sim_time = None
        self.last_elapsed_sim_sec = 0.0
        self.student_proposal_count = 0
        self._student_task_vector = None
        self._inference_latency_ms = []

class _InsertionState(_RunSharedPortTCP):

    def __init__(self, parent_node, checkpoint=None):
        super().__init__(parent_node)
        self.target_frame = 'estimated_port'
        self.visual_port = None
        self.visual_body_tip = None
        self.visual_body_connector_rotation = None
        self.local_terminal_reason = None
        self.robot_history = []
        self.governor = None
        self.governor_seed = None
        self.load_change_provider = None
        self.last_observation_stamp = None
        self.supervisor_callback = None

    def reset_for_task(self, task):
        super().reset_for_task(task)
        self.robot_history = []

    def retain_history_observation(self, observation):
        if observation is None or not self.control_metadata.use_local_history:
            return
        now = self._control_sim_time(observation)
        if self.robot_history and now < self.robot_history[-1][0]:
            raise ValueError('Observation clock reset inside local attempt')
        if not self.robot_history or now > self.robot_history[-1][0]:
            self.robot_history.append((now, base_state_from_ros_observation(observation).copy()))
        span = max(self.history_offsets_s) + 0.5
        while len(self.robot_history) > 1 and self.robot_history[1][0] < now - span:
            self.robot_history.pop(0)

    def _trace(self, **fields):
        if fields.get('event') == 'terminal_before_return':
            self.local_terminal_reason = fields.get('reason')
        return super()._trace(**fields)

    def bind_visual_handoff(self, report, sim_time):
        if not report['valid'] or report.get('runtime_true_geometry', True):
            raise ValueError('Only approved observation-only handoff can supply the frame')
        from scipy.spatial.transform import Rotation
        self.visual_port = np.eye(4)
        self.visual_port[:3, 3] = report['predicted_opening_base_m']
        self.visual_port[:3, :3] = Rotation.from_quat(report['predicted_port_quat_xyzw']).as_matrix()
        self.visual_lock_sim_time = sim_time

    def bind_predicted_body_tip(self, position):
        value = np.asarray(position, float)
        if value.shape != (3,) or not np.isfinite(value).all() or np.linalg.norm(value) > 0.1:
            raise ValueError('Invalid camera-estimated gripper-to-tip position')
        self.visual_body_tip = value

    def _control_sim_time(self, observation):
        stamp = super()._control_sim_time(observation)
        self.last_observation_stamp = stamp
        return stamp

    def insert_cable(self, *args, **kwargs):
        raise RuntimeError('Explicit insertion callback must be installed before execution')

class _ObservationContractError(Exception):
    pass

class _RecoveryTransfer(Exception):

    def __init__(self, decision):
        super().__init__(decision.reason or decision.mode)
        self.decision = decision

class _RunRuleThenSharedRecovery(Policy):

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.parent = parent_node
        self.local = self._make_local_actor(None)
        if set(self.local.trained_families) != {'sc', 'sfp'}:
            raise ValueError('This pipeline requires a shared actor trained on BOTH task families')
        self.mode = os.environ.get('AIC_SHARED_RECOVERY_MODE', 'hierarchical')
        if self.mode not in ('none', 'local', 'hierarchical'):
            raise ValueError('Unknown matched recovery arm')
        self.local.control_hz = 20.0
        self.local.control_clock = 'simulation'
        self.local.execute_steps = 4
        self.local.execute_complete_chunk = True
        self.local.max_translation = 0.06
        self.local.max_rotation = 0.15
        self.local.cartesian_stiffness = [90, 90, 90, 50, 50, 50]
        self.local.cartesian_damping = [50, 50, 50, 20, 20, 20]
        self.total_budget_s = float(os.environ.get('AIC_SHARED_RECOVERY_SIM_BUDGET_S', '180'))
        self.rule = None
        self.episode_index = 0
        self.perception_latencies_ms = []
        self.exploration_seed = int(os.environ.get('AIC_SHARED_RECOVERY_EXPLORATION_SEED', '20261016'))
        self._warm_actor()

    @torch.inference_mode()
    def _warm_grasp(self):
        self.grasp.model.eval()
        self.grasp.model(torch.zeros(3, 3, 192, 192, device=self.local.device), camera_id=torch.eye(3, device=self.local.device))
        if self.local.device.type == 'cuda':
            torch.cuda.synchronize(self.local.device)

    def _trace(self, **fields):
        self.local._trace(**fields)

    def _command(self, move_robot, matrix, *, owner):
        if hasattr(self.parent, '_target_mode'):
            self.parent._target_mode = TargetMode.MODE_CARTESIAN
        self.local.set_pose_target(move_robot=move_robot, pose=build_pose_from_vectors(matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()), frame_id='base_link', stiffness=self.local.cartesian_stiffness, damping=self.local.cartesian_damping)
        self._trace(event='supervisor_command', owner=owner, actor_owned=False, absolute_tcp_target=matrix.tolist(), geometry_input=False)

    def _force(self, state):
        return self.force.measure(state[26:29], state[3:7])['load_change_norm_n']

    def _initial_force_reference(self, get_observation, move_robot):
        self.force = InitialLoadReference(minimum_quiet_span_s=getattr(self, 'initial_force_minimum_quiet_span_s', 0.0))
        last = -float('inf')
        last_force = -float('inf')
        started = time.monotonic()
        hold = None
        while time.monotonic() - started < 20:
            observation = get_observation()
            if observation is None:
                time.sleep(0.005)
                continue
            sim = self.local._control_sim_time(observation)
            force_stamp = observation.wrist_wrench.header.stamp
            force_time = force_stamp.sec + force_stamp.nanosec * 1e-09
            if getattr(self, 'unique_initial_force_samples', False):
                if force_time <= last_force or abs(force_time - sim) > 0.1:
                    time.sleep(0.002)
                    continue
            elif sim - last < 0.05:
                time.sleep(0.002)
                continue
            state = self.local.assembler.assemble_ros(observation)
            if hold is None:
                hold = matrix_from_xyz_xyzw(state[:7])
            self._command(move_robot, hold, owner='initial_free_hold')
            last = sim
            if force_time <= last_force:
                continue
            last_force = force_time
            if self.force.add_initial_free_hold(stamp_s=force_time, force_sensor_n=state[26:29], tcp_quaternion_xyzw=state[3:7], linear_speed_mps=np.linalg.norm(state[7:10]), angular_speed_radps=np.linalg.norm(state[10:13])):
                self._trace(event='initial_force_reference_frozen', sim_time=sim, base_force_n=self.force.reference.tolist(), initial_free_hold_assumed=True, quiet_sample_span_s=self.force.timestamps[-1] - self.force.timestamps[0], quiet_samples_observed=len(self.force.observations), median_force_samples=self.force.samples, oracle_contact_force=False)
                return True
        return False

    def _lock(self, task_vector, get_observation, move_robot, *, fresh_context):
        self.local.clear_pending_actions()
        self.local.visual_port = None
        self.local.visual_body_tip = None
        self.local.robot_history = []
        if fresh_context:
            if task_vector[1] == 1:
                self.rule.verifier.reset_context()
            else:
                self.rule.observer.reset_context()
        previous = None
        predictions = []
        last = -float('inf')
        started = time.monotonic()
        while time.monotonic() - started < 20:
            observation = get_observation()
            if observation is None:
                time.sleep(0.005)
                continue
            sim = self.local._control_sim_time(observation)
            if sim - last < 0.15:
                time.sleep(0.002)
                continue
            last = sim
            inference_started = time.perf_counter()
            report, images, state, sim = self._fresh_report(observation, task_vector)
            self._command(move_robot, matrix_from_xyz_xyzw(self.acquisition_state[:7]), owner='fresh_visual_acquisition_hold')
            if not report['valid']:
                self.perception_latencies_ms.append((time.perf_counter() - inference_started) * 1000)
                previous = None
                predictions = []
                continue
            port = matrix_from_xyz_xyzw([*report['predicted_opening_base_m'], *report['predicted_port_quat_xyzw']])
            tcp = matrix_from_xyz_xyzw(state[:7])
            relative = np.linalg.inv(port) @ tcp
            gap = -relative[2, 3] * 1000
            if task_vector[0] == 1 and gap < 120:
                previous = None
                predictions = []
                continue
            if previous is not None:
                spread = np.linalg.norm(port[:3, 3] - previous[:3, 3])
                angle = (Rotation.from_matrix(previous[:3, :3]).inv() * Rotation.from_matrix(port[:3, :3])).magnitude()
                if spread > 0.001 or angle > np.deg2rad(1):
                    predictions = []
            lock_pose = port.copy()
            if previous is not None:
                lock_pose[:3, 3] = (port[:3, 3] + previous[:3, 3]) / 2
                lock_pose[:3, :3] = Rotation.from_matrix(np.stack([port[:3, :3], previous[:3, :3]])).mean().as_matrix()
            previous = port
            kwargs = dict(estimated_TCP_axial_gap_mm=float(gap)) if task_vector[0] == 1 else {}
            prediction = self.grasp.predict(images, **kwargs)
            duration = (time.perf_counter() - inference_started) * 1000
            self.perception_latencies_ms.append(duration)
            self._trace(event='shared_perception_latency', sim_time=sim, total_ms=duration, includes='three native RGB conversion + requested pose + body-tip estimate + hold publication')
            predictions = (predictions + [np.asarray(prediction['predicted_tip_position_tcp_m'])])[-3:]
            if len(predictions) < 3:
                continue
            if getattr(self, 'joint_grasp_acquisition', False) and (not report.get('connector_rotation_consensus_accepted', False)):
                continue
            required_far = getattr(self, 'required_SC_far_context_frames', 0) if task_vector[1] == 1 else 0
            if required_far and len(getattr(self, 'far_grasp_frames', [])) < required_far:
                self._trace(event='joint_acquisition_waiting_for_quiet_far_context', sim_time=sim, available_frames=len(getattr(self, 'far_grasp_frames', [])), required_frames=required_far, geometry_input=False)
                continue
            body = np.median(predictions, axis=0)
            spread = max((np.linalg.norm(x - body) for x in predictions))
            if spread > 0.0005:
                continue
            locked_relative = np.linalg.inv(lock_pose) @ tcp
            tip = locked_relative[:3, 3] + locked_relative[:3, :3] @ body
            minimum_tip_gap = getattr(self, 'SFP_minimum_shared_tip_standoff_m', 0.1) if task_vector[0] == 1 else 0.1
            if not 0.05 <= minimum_tip_gap <= 0.1:
                raise ValueError('Invalid explicit SFP acquisition envelope')
            maximum_tip_lateral = getattr(self, 'SFP_maximum_shared_tip_lateral_m', 0.03) if task_vector[0] == 1 else 0.03
            if not 0.03 <= maximum_tip_lateral <= 0.06:
                raise ValueError('Invalid explicit SFP lateral acquisition envelope')
            if not (-0.185 <= tip[2] <= -minimum_tip_gap and np.linalg.norm(tip[:2]) <= maximum_tip_lateral):
                self._trace(event='shared_lock_rejected', reason='outside_approach_envelope', tip_port_m=tip.tolist())
                return False
            report = dict(report, predicted_opening_base_m=lock_pose[:3, 3].tolist(), predicted_port_quat_xyzw=Rotation.from_matrix(lock_pose[:3, :3]).as_quat().tolist(), lock_measurements=2, lock_rule='same two-consistent-pose mean as BC frame preparation')
            self.local.bind_visual_handoff(report, sim)
            self.local.bind_predicted_body_tip(body)
            self.acquisition_state = state.copy()
            self._trace(event='shared_visual_lock', sim_time=sim, report=report, body_tip_position_tcp_m=body.tolist(), grasp_spread_mm=float(spread * 1000), requested_task=np.asarray(task_vector).tolist(), fresh_context=fresh_context, geometry_input=False, actor_owns_full_150mm=True)
            return True
        return False

    def _supervise(self, state, absolute):
        force = self._force(state)
        relative = np.linalg.inv(self.local.visual_port) @ matrix_from_xyz_xyzw(state[:7])
        tip = relative[:3, 3] + relative[:3, :3] @ self.local.visual_body_tip
        phase = 'contact' if tip[2] > -0.005 else 'alignment' if tip[2] > -0.04 else 'approach'
        demand = 0.0 if self.last_target is None else float(np.linalg.norm(self.last_target - state[:3]))
        decision = self.supervisor.step(pose=measured_pose(state), force_n=force, prior_command_m=self.last_command_m, tracking_error_m=demand, phase=phase, route=self.route)
        self._trace(event='recovery_observation', mode=decision.mode, level=decision.level, measured_state=state.tolist(), load_change_n=force, prior_command_m=self.last_command_m, tracking_error_m=demand, phase=phase, route=self.route, actor_owned=decision.actor_owned)
        if not decision.actor_owned:
            self.local.clear_pending_actions()
            raise _RecoveryTransfer(decision)
        self.last_target = absolute[:3, 3].copy()
        self.last_command_m = float(np.linalg.norm(absolute[:3, 3] - state[:3]))

    def _retreat(self, transfer, get_observation, move_robot):
        if transfer.decision.mode == 'abort':
            return False
        route_level = transfer.decision.level == 'route'
        self.blocked_pose = self.supervisor.previous_pose
        if route_level:
            if not self.supervisor.extend_route_to_acquisition():
                return False
            self.local.visual_port = None
            self.local.visual_body_tip = None
        else:
            inverse = np.linalg.inv(self.local.visual_port)
            body = self.local.visual_body_tip

            def outside(p):
                q = p.quat_wxyz
                m = matrix_from_xyz_xyzw([*p.xyz, q[1], q[2], q[3], q[0]])
                return (inverse @ np.r_[m[:3, 3] + m[:3, :3] @ body, 1.0])[2] <= -0.008
            if not outside(self.supervisor.anchor):
                candidates = [i for i in range(self.supervisor.anchor_history_index, -1, -1) if self.supervisor.history[i][1] <= self.supervisor.config.clear_force_n and outside(self.supervisor.history[i][0])]
                if not candidates:
                    self._trace(event='recovery_abort', reason='no_measured_outside_opening_checkpoint')
                    return False
                i = candidates[0]
                self.supervisor.path = [x[0] for x in reversed(self.supervisor.history[i:])]
                self.supervisor.path_index = 0
                self.supervisor.anchor = self.supervisor.history[i][0]
                self.supervisor.anchor_history_index = i
        started = time.monotonic()
        last = -float('inf')
        while time.monotonic() - started < 60:
            observation = get_observation()
            if observation is None:
                time.sleep(0.005)
                continue
            sim = self.local._control_sim_time(observation)
            if sim - self.insertion_start_sim > self.total_budget_s:
                return False
            if sim - last < 0.05:
                time.sleep(0.002)
                continue
            last = sim
            state = self.local.assembler.assemble_ros(observation)
            decision = self.supervisor.step(pose=measured_pose(state), force_n=self._force(state), prior_command_m=0.0, tracking_error_m=0.0, phase='recovery', route=self.route)
            if decision.mode == 'abort':
                self._trace(event='recovery_abort', reason=decision.reason, sim_time=sim)
                return False
            if decision.mode == 'retry':
                self._trace(event='measured_retreat_complete', sim_time=sim, level=decision.level, total_backoffs=decision.total_backoffs, route_resets=decision.route_resets)
                return True
            if decision.target is not None:
                p = decision.target
                q = p.quat_wxyz
                target = matrix_from_xyz_xyzw([*p.xyz, q[1], q[2], q[3], q[0]])
                self._command(move_robot, target, owner='measured_path_backtrack')
        self._trace(event='recovery_abort', reason='retreat_timeout')
        return False

    def insert_cable(self, task, get_observation, move_robot, send_feedback, **kwargs):
        self.episode_index += 1
        self.local.reset_for_task(task)
        self.local.clear_pending_actions()
        self.local.visual_port = None
        self.local.visual_body_tip = None
        self.local.local_terminal_reason = None
        self.local.supervisor_callback = None
        self.rule = None
        self.episode_deadline = time.monotonic() + max(1.0, float(task.time_limit) - 5.0)
        self.perception_latencies_ms = []
        self.pipeline_phase = 'initializing'
        self.pipeline_outcome = False
        self.retained_terminal = None
        self.retained_native = None
        self.retained_wall = None

        def checked_observation():
            observation = get_observation()
            if observation is not None:
                sim = self.local._control_sim_time(observation)
                if self.retained_terminal is None or sim > self.retained_terminal['sim_time']:
                    self.retained_terminal = dict(sim_time=sim, observation_wall_time_ns=time.time_ns(), state=self.local.assembler.assemble_ros(observation).tolist())
                    self.retained_native = observation
                    self.retained_wall = time.monotonic()
                header = observation.wrist_wrench.header
                force_time = header.stamp.sec + header.stamp.nanosec * 1e-09
                if header.frame_id.lstrip('/') != 'ati/tool_link' or abs(sim - force_time) > 0.1:
                    self.local.clear_pending_actions()
                    self._trace(event='terminal_before_return', reason='invalid_or_stale_wrist_force', sim_time=sim, state=self.local.assembler.assemble_ros(observation).tolist(), wrench_frame=header.frame_id, wrench_age_s=sim - force_time)
                    raise _ObservationContractError()
            if time.monotonic() >= self.episode_deadline:
                self.local.clear_pending_actions()
                self._trace(event='terminal_before_return', reason='whole_task_budget', sim_time=self.local._control_sim_time(observation) if observation is not None else None, state=self.local.assembler.assemble_ros(observation).tolist() if observation is not None else None, post_reset_geometry_used=False)
                raise _RuntimeBudgetReached()
            return observation
        try:
            self.pipeline_outcome = self._execute(task, checked_observation, move_robot, send_feedback, **kwargs)
            return self.pipeline_outcome
        except (_RuntimeBudgetReached, _ObservationContractError):
            return False
        except Exception as error:
            self._trace(event='pipeline_exception', stage=self.pipeline_phase, error_type=type(error).__name__, message=str(error), implementation_failure=True, geometry_input=False)
            raise
        finally:
            self.local.clear_pending_actions()
            terminal_images = {}
            image_stamps = {}
            if self.retained_native is not None and self.local.trace_path:
                folder = Path(self.local.trace_path).parent / 'terminal_native_rgb' / f'action_{self.episode_index:05d}'
                folder.mkdir(parents=True, exist_ok=False)
                for camera in ('center', 'left', 'right'):
                    message = getattr(self.retained_native, camera + '_image')
                    image = _RunPredictedPortLocal.raw_rgb(message)
                    path = folder / (camera + '.jpg')
                    if cv2.imwrite(str(path), image[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 86]):
                        terminal_images[camera] = str(path)
                    image_stamps[camera] = message.header.stamp.sec * 1000000000 + message.header.stamp.nanosec
            self._trace(event='terminal_before_return', reason='shared_pipeline_' + self.pipeline_phase, **self.retained_terminal or dict(sim_time=None, state=None), observation_source='retained_public_observation_before_return', observation_age_wall_s=time.monotonic() - self.retained_wall if self.retained_wall is not None else None, terminal_images=terminal_images, image_stamps_ns=image_stamps, attempt_acknowledged=bool(self.pipeline_outcome), physical_insertion_asserted=False, underlying_actor_terminal_reason=self.local.local_terminal_reason, visual_port_matrix=self.local.visual_port.tolist() if self.local.visual_port is not None else None, body_tip_position_tcp_m=self.local.visual_body_tip.tolist() if self.local.visual_body_tip is not None else None, post_reset_geometry_used=False, task_id=str(task.id), episode_index=self.episode_index)
            if self.perception_latencies_ms:
                self._trace(event='shared_perception_latency_summary', calls=len(self.perception_latencies_ms), p50_p95_p99_ms=np.percentile(self.perception_latencies_ms, [50, 95, 99]).tolist())

    def _warm_actor(self):
        return None

class _RunSampleThenSharedInsertion(_RunRuleThenSharedRecovery):

    def _fresh_report(self, observation, task_vector):
        state = self.local.assembler.assemble_ros(observation)
        sim = self.local._control_sim_time(observation)
        images = {}
        if getattr(self, 'parallel_fresh_JPEG', False):
            from .runtime.parallel_training_jpeg import normalise_triplet
            images = normalise_triplet({camera: _RunRuleTransport.native(getattr(observation, camera + '_image')) for camera in ('center', 'left', 'right')})
        else:
            for camera in ('center', 'left', 'right'):
                rgb = _RunRuleTransport.native(getattr(observation, camera + '_image'))
                ok, encoded = cv2.imencode('.jpg', rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 86])
                if not ok:
                    raise RuntimeError('Native RGB codec failed')
                images[camera] = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        prediction = self.rule.component.predict(images, state, task_vector)
        report = dict(valid=False, reason=prediction['reason'], runtime_true_geometry=False)
        if prediction['accepted']:
            port = np.asarray(prediction['base_port_matrix'])
            envelope = 0.35 if task_vector[0] == 1 else 0.2
            within = np.linalg.norm(port[:3, 3] - self.rule.nominal_port[:3, 3]) <= envelope
            report.update(valid=bool(within), reason='fresh_requested_native_pose' if within else 'outside_public_sample_search_envelope', predicted_opening_base_m=port[:3, 3].tolist(), predicted_port_quat_xyzw=Rotation.from_matrix(port[:3, :3]).as_quat().tolist())
        return (report, images, state, sim)

    def _execute(self, task, get_observation, move_robot, send_feedback, **kwargs):
        self.exploration_rng = np.random.default_rng(self.exploration_seed)
        task_vector = self.local.task_vector(task)
        self.rule = self._make_rule_transport()
        if task_vector[1] == 1:
            self.grasp = SCNativeGraspObservation(os.environ['AIC_LOCKED_GRASP_CHECKPOINT'], str(self.local.device))
        else:
            self.grasp = SFPNativeGraspObservation(os.environ['AIC_SFP_GRASP_CHECKPOINT'], os.environ['AIC_SFP_GRASP_APPROVAL'], str(self.local.device))
        self._warm_grasp()
        self.pipeline_phase = 'initial_force_reference'
        if not self._initial_force_reference(get_observation, move_robot):
            return False
        self.pipeline_phase = 'rule_transport'
        self.rule.insert_cable(task, get_observation, move_robot, send_feedback, **kwargs)
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
        self.supervisor = HierarchicalRecovery(RecoveryConfig(max_total_backoffs=8, max_history=4096, local_attempts_before_route=2 if self.mode == 'hierarchical' else 100))
        self.last_target = None
        self.last_command_m = 0.0
        self.route = 'acquisition_0'
        self.supervisor.step(pose=measured_pose(self.acquisition_state), force_n=self._force(self.acquisition_state), prior_command_m=0.0, phase='acquisition_checkpoint', route=self.route)
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
            self.local.max_runtime_sec = remaining_wall - 1.0
            self.pipeline_phase = 'actor'
            try:
                return self.local.insert_cable(task, get_observation, move_robot, send_feedback, **kwargs)
            except _RecoveryTransfer as transfer:
                self.last_target = None
                self.last_command_m = 0.0
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
                    self.route = self.supervisor.confirm_new_corridor(f'acquisition_{self.supervisor.route_resets}', self.last_measured_lateral_retry_m)
                self.local.clear_pending_actions()
                self.local.robot_history = []

class _RunSample25mmThenInsertion(_RunSampleThenSharedInsertion):

    def __init__(self, parent_node):
        super().__init__(parent_node)
        if self.mode != 'none':
            raise ValueError('The25mm handoff diagnostic requires recovery=none; preserve old hierarchical controls')

    def _approach_orientation_tolerance(self):
        return np.deg2rad(2.0)

    def _lock(self, task_vector, get_observation, move_robot, *, fresh_context):
        if not super()._lock(task_vector, get_observation, move_robot, fresh_context=fresh_context):
            return False
        port = self.local.visual_port.copy()
        inverse = np.linalg.inv(port)
        body = self.local.visual_body_tip.copy()
        rotation = self._approach_tcp_rotation(port)
        integral = np.zeros(3)
        tracker = None
        orientation_tracker = None
        reference = None
        last = None
        first = None
        best = float('inf')
        last_progress = None
        stable_since = None
        force_streak = 0
        started = time.monotonic()
        self.pipeline_phase = 'visual_25mm_approach'
        while time.monotonic() - started < 100:
            observation = get_observation()
            if observation is None:
                time.sleep(0.002)
                continue
            sim = self.local._control_sim_time(observation)
            if last is not None and sim - last < 0.049:
                time.sleep(0.002)
                continue
            dt = 0.05 if last is None else min(0.1, sim - last)
            last = sim
            state = self.local.assembler.assemble_ros(observation)
            measured = matrix_from_xyz_xyzw(state[:7])
            point = (inverse @ np.r_[measured[:3, 3] + measured[:3, :3] @ body, 1.0])[:3]
            angle = (Rotation.from_matrix(measured[:3, :3]).inv() * Rotation.from_matrix(rotation)).magnitude()
            if first is None:
                first = sim
                reference = np.array([0.0, 0.0, point[2]])
                if getattr(self, 'approach_antiwindup', False):
                    from .runtime.approach_reference import ApproachReference
                    bias_base = state[13:16]
                    if getattr(self, 'approach_bias_from_issued_reference', False):
                        bias_base = self.local.governor_seed[0][:3, 3] - measured[:3, 3]
                    tracker = ApproachReference(point, inverse[:3, :3] @ bias_base)
                    reference = tracker.reference.copy()
                    integral = tracker.integral.copy()
                    self._trace(event='approach_public_bias_seeded', sim_time=sim, tracking_bias_port_m=integral.tolist(), geometry_input=False)
                if getattr(self, 'persistent_approach_orientation', False):
                    from .runtime.approach_orientation_reference import ApproachOrientationReference
                    seed = self.local.governor_seed[0]
                    orientation_tracker = ApproachOrientationReference(seed[:3, :3], rotation)
                last_progress = sim
            distance = np.linalg.norm(point - [0.0, 0.0, -0.025])
            if distance < best - 0.0005:
                best = distance
                last_progress = sim
            force = self._force(state)
            force_streak = force_streak + 1 if force >= 8 else 0
            failure = 'raw_force_guard' if np.linalg.norm(state[26:29]) > 40 else 'load_change_guard' if force_streak >= 2 else 'approach_simulation_budget' if sim - first > 55 else 'approach_no_progress' if sim - last_progress > 12 else None
            if failure:
                self._trace(event='near_handoff_failed', sim_time=sim, reason=failure, state=state.tolist(), estimated_tip_port_m=point.tolist(), load_change_n=force, geometry_input=False, actor_owned=False)
                return False
            alignment_limit = 0.003 if point[2] < -0.05 else 0.0015
            aligned = np.linalg.norm(point[:2]) <= alignment_limit and angle <= self._approach_orientation_tolerance()
            if tracker is not None:
                if point[2] > -0.02:
                    self._trace(event='near_handoff_failed', sim_time=sim, reason='approach_crossed20mm_buffer', geometry_input=False)
                    return False
                reference, integral = tracker.step(point, inverse[:3, :3] @ state[7:10], dt, self._approach_speed(point[2]) if aligned else 0.0)
            else:
                if aligned:
                    speed = self._approach_speed(point[2])
                    reference[2] = min(-0.025, reference[2] + speed * dt)
                integral = np.clip(integral + 0.6 * dt * (reference - point), -0.015, 0.015)
            commanded_point = reference + integral
            goal = measured.copy()
            relative_rotation = Rotation.from_matrix(measured[:3, :3]).inv() * Rotation.from_matrix(rotation)
            rotvec = relative_rotation.as_rotvec()
            rotvec *= min(1.0, np.deg2rad(1.0) / max(np.linalg.norm(rotvec), 1e-12))
            angular_diagnostic = None
            if orientation_tracker is None:
                goal[:3, :3] = measured[:3, :3] @ Rotation.from_rotvec(rotvec).as_matrix()
            else:
                goal[:3, :3], angular_diagnostic = orientation_tracker.step(measured[:3, :3], dt)
            destination = (port @ np.r_[commanded_point, 1.0])[:3] - goal[:3, :3] @ body
            step = destination - measured[:3, 3]
            goal[:3, 3] += step * min(1.0, 0.02 / max(np.linalg.norm(step), 1e-12))
            self._command(move_robot, goal, owner='visual_transport_150_to25mm')
            self._trace(event='near_visual_approach', sim_time=sim, state=state.tolist(), estimated_tip_port_m=point.tolist(), reference_tip_port_m=reference.tolist(), commanded_tip_port_m=commanded_point.tolist(), integral_port_m=integral.tolist(), antiwindup=tracker is not None, load_change_n=force, desired_rotation_error_deg=float(np.rad2deg(angle)), persistent_orientation_reference=angular_diagnostic, port_source='fresh_RGB_lock_at150mm_rigid_fixture', geometry_input=False, actor_owned=False)
            ready = abs(point[2] + 0.025) <= 0.002 and np.linalg.norm(point[:2]) <= 0.001 and (angle <= self._approach_orientation_tolerance()) and (np.linalg.norm(state[7:10]) <= 0.005)
            stable_since = (sim if stable_since is None else stable_since) if ready else None
            if stable_since is not None and sim - stable_since >= 0.5:
                self.acquisition_state = state.copy()
                self.local.robot_history = []
                self._trace(event='near_handoff_ready', sim_time=sim, state=state.tolist(), estimated_tip_port_m=point.tolist(), desired_gap_mm=25.0, estimated_orientation_error_deg=float(np.rad2deg(angle)), orientation_tolerance_deg=float(np.rad2deg(self._approach_orientation_tolerance())), geometry_input=False, learned_actor_not_yet_started=True, observation_source='retained_public_observation_before_return', observation_wall_time_ns=time.time_ns(), post_reset_geometry_used=False)
                return True
        self._trace(event='near_handoff_failed', reason='approach_wall_budget', geometry_input=False)
        return False

class _RunHistoryChunkedInsertion(_RunSample25mmThenInsertion):

    def _fresh_report(self, observation, task_vector):
        report, _, state, sim = super()._fresh_report(observation, task_vector)
        raw = {c: _RunRuleTransport.native(getattr(observation, c + '_image')) for c in ('center', 'left', 'right')}
        if report['valid']:
            pose = matrix_from_xyz_xyzw([*report['predicted_opening_base_m'], *report['predicted_port_quat_xyzw']])
            accepted = self.pose_consensus.update(pose, sim)
            if accepted is None:
                report = dict(report, valid=False, reason='awaiting_fresh_pose_window_consensus')
            else:
                pose = accepted.pop('matrix')
                report = dict(report, predicted_opening_base_m=pose[:3, 3].tolist(), predicted_port_quat_xyzw=Rotation.from_matrix(pose[:3, :3]).as_quat().tolist(), acquisition_consensus=accepted)
        else:
            self.pose_consensus.clear()
        return (report, raw, state, sim)

    def _lock(self, task_vector, get_observation, move_robot, *, fresh_context):
        self.pose_consensus.clear()
        return super()._lock(task_vector, get_observation, move_robot, fresh_context=fresh_context)

    def insert_cable(self, task, get_observation, move_robot, send_feedback, **kwargs):
        self.history_task_id = str(task.id)
        self.observation_history = []

        def observed():
            value = get_observation()
            if value is not None:
                stamp = self.local._control_sim_time(value)
                if self.observation_history and stamp < self.observation_history[-1][0] - 1e-06:
                    raise _ObservationContractError('Clock reset inside the active history window')
                if not self.observation_history or stamp > self.observation_history[-1][0]:
                    state = self.local.assembler.assemble_ros(value)
                    if len(state) != 32 or not np.isfinite(state).all():
                        raise _ObservationContractError('Invalid public robot state')
                    self.observation_history.append((stamp, state.copy()))
                    span = max(self.local.history_offsets_s or [1.2]) + 0.5
                    while len(self.observation_history) > 1 and self.observation_history[1][0] < stamp - span:
                        self.observation_history.pop(0)
            return value
        return super().insert_cable(task, observed, move_robot, send_feedback, **kwargs)

    def __init__(self, parent_node):
        self.history_task_id = None
        self.observation_history = []
        self.pose_consensus = StablePoseWindow()
        super().__init__(parent_node)
        self.local.execute_steps = 4

class _RunFullConnectorInsertion(_RunHistoryChunkedInsertion):

    def __init__(self, parent_node):
        super().__init__(parent_node)
        if self.local.sc_action_reference != 'full_connector_pose' or self.local.tip_input_mode != 'full_pose':
            raise ValueError('This diagnostic requires the exact full-connector representation')
        self.connector_models = {}
        self.connector_rotation_window = []
        self.locked_connector_rotation = None

    def _fresh_report(self, observation, task_vector):
        report, images, state, sim = super()._fresh_report(observation, task_vector)
        if not report['valid']:
            self.connector_rotation_window = []
            return (report, images, state, sim)
        family = 'sfp' if task_vector[0] == 1 else 'sc'
        if family not in self.connector_models:
            filename = ROOT / ('sfp_direct_pose' if family == 'sfp' else 'sc_orientation_jitter_v2') / 'best.pt'
            self.connector_models[family] = NativeConnectorPoseObservation(filename, str(self.local.device))
        started = time.perf_counter()
        estimate = self.connector_models[family].predict(images, visual_tip_position_tcp_m=np.zeros(3))
        rotation = np.asarray(estimate['orientation_for_diagnostic_only'])
        self.connector_rotation_window = [x for x in self.connector_rotation_window if 0 <= sim - x[0] <= 1.0]
        if self.connector_rotation_window and sim <= self.connector_rotation_window[-1][0]:
            return (dict(report, valid=False, reason='repeated_orientation_timestamp'), images, state, sim)
        self.connector_rotation_window = (self.connector_rotation_window + [(sim, rotation)])[-3:]
        accepted = False
        span = None
        if len(self.connector_rotation_window) == 3:
            rotations = [x[1] for x in self.connector_rotation_window]
            span = max((float((Rotation.from_matrix(a).inv() * Rotation.from_matrix(b)).magnitude() * 180 / np.pi) for a in rotations for b in rotations))
            if span <= 1.0:
                self.locked_connector_rotation = Rotation.from_matrix(np.stack(rotations)).mean().as_matrix()
                accepted = True
            else:
                self.connector_rotation_window = self.connector_rotation_window[1:]
        self._trace(event='connector_rotation_acquisition', sim_time=sim, family=family, rotation_tcp=rotation.tolist(), window_samples=len(self.connector_rotation_window), accepted=accepted, span_deg=span, orientation_model_ms=(time.perf_counter() - started) * 1000, geometry_input=False, position_placeholder_not_used_for_control=True)
        report = dict(report, connector_rotation_consensus_accepted=bool(accepted))
        if not accepted and (not getattr(self, 'joint_grasp_acquisition', False)):
            report = dict(report, valid=False, reason='awaiting_stable_connector_rotation')
        return (report, images, state, sim)

    def _lock(self, task_vector, get_observation, move_robot, *, fresh_context):
        self.connector_rotation_window = []
        self.locked_connector_rotation = None
        self.local.visual_body_connector_rotation = None
        if not super()._lock(task_vector, get_observation, move_robot, fresh_context=fresh_context):
            return False
        if self.locked_connector_rotation is None:
            raise ValueError('Missing causal connector orientation')
        self.local.visual_body_connector_rotation = self.locked_connector_rotation.copy()
        self._trace(event='full_connector_grasp_bound', rotation_tcp=self.locked_connector_rotation.tolist(), position_tcp_m=self.local.visual_body_tip.tolist(), source='three native RGB frames before approach', geometry_input=False, static_within_attempt=True)
        return True

class _RunContinuousConnectorInsertion(_RunFullConnectorInsertion):

    def _approach_tcp_rotation(self, port):
        if self.locked_connector_rotation is None:
            raise ValueError('Full RGB grasp rotation must precede approach')
        return port[:3, :3] @ self.locked_connector_rotation.T

    def _command(self, move_robot, matrix, *, owner):
        super()._command(move_robot, matrix, owner=owner)
        if owner == 'visual_transport_150_to25mm':
            self.local.governor = None
            self.local.governor_seed = (np.asarray(matrix).copy(), self.local.last_observation_stamp)

class _RunContinuousConnectorInsertionV2(_RunContinuousConnectorInsertion):

    def _make_local_actor(self, checkpoint=None):
        state = _InsertionState(self.parent)
        state.history_provider = lambda: dict(task_id=self.history_task_id, rows=self.observation_history)
        state.load_change_provider = lambda values: self._force(values)
        return state

class _RunSampleConservativeTransport(_RunSharedPortTCP):

    def __init__(self, parent_node):
        torch.set_num_threads(2)
        cv2.setNumThreads(1)
        super().__init__(parent_node)
        self.spec = json.loads(Path(os.environ['AIC_RULE_TRANSPORT_TEMPLATE']).read_text())
        if self.spec.get('schema') != 'sample_conservative_transport/v1' or self.spec['geometry_runtime_input']:
            raise ValueError('Pinned public-sample specification required')
        self.target_frame = 'body'
        self.execute_steps = 4
        self.transport_terminal_reason = 'sample_rule_acquisition_or_guarded_failure'
        self.components = {}
        self._reset()

    def _reset(self):
        self.family = None
        self.params = None
        self.component = None
        self.stages = None
        self.index = 0
        self.reference = None
        self.compensation = np.zeros(3)
        self.last_inspection = -float('inf')
        self.pending = None
        self.pose = None
        self.pose_time = -float('inf')
        self.pose_verified_time = -float('inf')
        self.streak = 0
        self.rule_failure = None
        self.rule_final_valid = False
        self.report = None
        self.hold = None
        self.last_progress = None
        self.best_error = float('inf')
        self.stage_streak = 0
        self.acquisition_start = None
        self.scan_index = 0
        self.scan_since = None
        self.marker_pending = None
        self.marker_count = 0
        self.marker_routed = False
        self.marker_started = None

    def reset_for_task(self, task):
        super().reset_for_task(task)
        self._reset()
        for c in self.components.values():
            c.reset()

    def _setup(self, task, current, sim):
        self.family = 'sfp' if task[0] > 0.5 else 'sc'
        self.params = self.spec['families'][self.family]
        b = self.spec['bundle']
        if self.family not in self.components:
            self.components[self.family] = RequestedNativePortPose(b['backbone'], b[self.family + '_coarse'], b[self.family + '_native_refiner'], b['mount_priors'], family=self.family, device=str(self.device), maximum_shift_px=64.0, maximum_reprojection_px=2.5 if self.family == 'sc' else 2.0, minimum_projection_probability=0.9 if self.family == 'sc' else 0.95)
        self.component = self.components[self.family]
        port = int(np.argmax(task[2:4]))
        card = int(np.argmax(task[4:9])) if self.family == 'sfp' else None
        self.requested_port = port
        self.requested_card = card
        self.nominal_port = np.asarray(self.params['nominal_ports_base'][str(port) if card is None else f'{card}:{port}'])
        self.reference = current.copy()
        self.last_progress = sim
        self._build_stages(current, self.nominal_port)

    def _tcp_goal(self, port):
        goal = np.eye(4)
        goal[:3, :3] = port[:3, :3] @ Rotation.from_quat(self.params['nominal_tcp_in_port_xyzw']).as_matrix()
        if self.family == 'sc':
            goal[:3, 3] = port[:3, 3] + self.params['sc_opening_to_tcp_base_m']
        else:
            goal[:3, 3] = port[:3, 3] + port[:3, :3] @ np.array([0.0, 0.0, -0.15]) - goal[:3, :3] @ self.params['body_tip_offset_m']
        return goal

    def _build_stages(self, current, port):
        final = self._tcp_goal(port)
        board = np.asarray(self.params['nominal_board_base'])
        if self.family == 'sc':
            p = self.params['bypass']
            initial = current[:3, :3]
            self.route_initial_rotation = initial.copy()

            def point(x, y, z):
                v = board @ np.array([x, y, 0.0, 1.0])
                v[2] = z
                return v[:3]
            start_local = np.linalg.inv(board) @ np.r_[current[:3, 3], 1.0]
            final_local = np.linalg.inv(board) @ np.r_[final[:3, 3], 1.0]
            behind = p['behind_board_y_m']
            beyond = p['beyond_board_y_m']
            outside = p['outside_board_x_m']
            low = p['low_tcp_z_m']
            route = os.environ.get('AIC_SC_TRANSPORT_ROUTE', 'sc_old')
            if route == 'sc_old':
                entry = [('clear_start_side', point(start_local[0], behind, max(low + 0.05, current[2, 3] - 0.03)), initial), ('shift_behind_row_low', point(outside, behind, low), initial)]
            elif route == 'sc_skip_back':
                entry = [('shift_from_start_low', point(outside, start_local[1], low), initial)]
            elif route == 'sc_level_shift':
                entry = [('shift_from_start_level', point(outside, start_local[1], float(current[2, 3])), initial), ('descend_outside_row', point(outside, start_local[1], low), initial)]
            elif route == 'sc_hover':
                entry = [('shift_from_start_level', point(outside, start_local[1], float(current[2, 3])), initial)]
            elif route == 'sc_marker_scout':
                look = initial @ Rotation.from_rotvec([0.25, -0.4, 0.08]).as_matrix()
                entry = [('shift_from_start_level', point(outside, start_local[1], float(current[2, 3])), initial), ('look_for_board_marker', point(outside, start_local[1], float(current[2, 3])), look)]
            elif route == 'sc_high_shift':
                raised = max(current[2, 3] + 0.05, low + 0.12)
                entry = [('lift_before_side', point(start_local[0], start_local[1], raised), initial), ('shift_from_start_high', point(outside, start_local[1], raised), initial), ('descend_outside_row', point(outside, start_local[1], low), initial)]
            elif route == 'sc_ascend':
                raised = current[2, 3] + 0.1
                entry = [('ascend_before_back', point(start_local[0], start_local[1], raised), initial), ('clear_start_side_high', point(start_local[0], behind, raised), initial), ('shift_behind_row_low', point(outside, behind, low), initial)]
            else:
                raise ValueError(f'Unknown SC transport route: {route}')
            self._trace(event='sc_transport_route_selected', route=route, start_board_local_m=start_local[:3].tolist(), geometry_input=False)
            early_tilt = os.environ.get('AIC_SC_EARLY_TILT', '0') == '1'
            travel_z = float(current[2, 3]) if route in ('sc_hover', 'sc_marker_scout') else low
            if early_tilt:
                finish = [('tilt_outside_row', point(outside, beyond, travel_z), final[:3, :3]), ('sweep_after_row', point(final_local[0], beyond, travel_z), final[:3, :3])]
            else:
                finish = [('sweep_after_row', point(final_local[0], beyond, travel_z), initial), ('tilt_clear_of_cards', point(final_local[0], beyond, travel_z), final[:3, :3])]
            self._trace(event='sc_transport_tilt_order_selected', early_tilt=early_tilt, geometry_input=False)
            self.stages = entry + [('traverse_beyond_row', point(outside, beyond, travel_z), initial), *finish, ('visual_acquisition', final[:3, 3], final[:3, :3])]
        else:
            z = max(current[2, 3], final[2, 3] + 0.05)
            a = current[:3, 3].copy()
            a[2] = z
            b = final[:3, 3].copy()
            b[2] = z
            self.stages = [('lift_clear_of_cards', a, current[:3, :3]), ('transfer_above_row', b, current[:3, :3]), ('orient_clear_of_cards', b, final[:3, :3]), ('visual_acquisition', final[:3, 3], final[:3, :3])]

    def _inspect(self, observation, state, task, sim):
        images = {c: _RunRuleTransport.native(getattr(observation, c + '_image')) for c in ['center', 'left', 'right']}
        if getattr(self, 'parallel_transport_JPEG', False):
            from .runtime.parallel_training_jpeg import normalise_triplet
            images = normalise_triplet(images)
        else:
            for c, image in images.items():
                ok, jpg = cv2.imencode('.jpg', image[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 86])
                if not ok:
                    raise RuntimeError('Native RGB codec failure')
                images[c] = cv2.cvtColor(cv2.imdecode(jpg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        result = self.component.predict(images, state, task)
        candidate = np.asarray(result['base_port_matrix']) if result['accepted'] else None
        reason = result['reason']
        valid = False
        if candidate is not None:
            nominal = self.nominal_port[:3, 3]
            if np.linalg.norm(candidate[:3, 3] - nominal) > (0.35 if self.family == 'sfp' else 0.2):
                candidate = None
                reason = 'outside_public_sample_search_envelope'
        if candidate is not None:
            if self.pose is not None and (np.linalg.norm(candidate[:3, 3] - self.pose[:3, 3]) > 0.005 or (Rotation.from_matrix(candidate[:3, :3]).inv() * Rotation.from_matrix(self.pose[:3, :3])).magnitude() > np.deg2rad(5.0)):
                self.pose = None
                self.pose_verified_time = -float('inf')
            agrees = self.pending is not None and np.linalg.norm(candidate[:3, 3] - self.pending[:3, 3]) <= 0.001 and ((Rotation.from_matrix(candidate[:3, :3]).inv() * Rotation.from_matrix(self.pending[:3, :3])).magnitude() <= np.deg2rad(1.0))
            self.streak = self.streak + 1 if agrees else 1
            self.pending = candidate
            if self.streak >= 2:
                self.pose = candidate
                self.pose_time = sim
            if self.streak >= 3:
                self.pose_verified_time = sim
            goal = self._tcp_goal(candidate)
            error = float(np.linalg.norm(goal[:3, 3] - state[:3]))
            angle = float((Rotation.from_matrix(goal[:3, :3]).inv() * Rotation.from_quat(state[3:7])).magnitude())
            valid = bool(self.streak >= 3 and error <= 0.015 and (angle <= np.deg2rad(3.0)))
            reason = 'clear_selected_port_and_safe_standoff' if valid else 'not_at_selected_standoff'
        else:
            self.streak = 0
        self.report = dict(valid=valid, reason=reason, requested_port=self.requested_port, requested_card=self.requested_card, predicted_opening_base_m=candidate[:3, 3].tolist() if candidate is not None else None, predicted_port_quat_xyzw=Rotation.from_matrix(candidate[:3, :3]).as_quat().tolist() if candidate is not None else None, native_pose_component=result, runtime_true_geometry=False)
        event = 'SFP_rule_visual_inspection' if self.family == 'sfp' else 'rule_visual_inspection'
        self._trace(event=event, sim_time=sim, report=self.report, geometry_input=False)
        if self.family == 'sc' and os.environ.get('AIC_SC_TRANSPORT_ROUTE') == 'sc_marker_scout' and (self.index == 1) and (not self.marker_routed):
            from .runtime.observed_board_marker import estimate_board_marker
            estimate = estimate_board_marker(images, state, self.component.optics, self.params['nominal_board_base'], self.nominal_port)
            if estimate is not None:
                board = estimate['board']
                agrees = self.marker_pending is not None and np.linalg.norm(board[:3, 3] - self.marker_pending[:3, 3]) <= 0.02 and ((Rotation.from_matrix(board[:3, :3]).inv() * Rotation.from_matrix(self.marker_pending[:3, :3])).magnitude() <= np.deg2rad(3.0))
                self.marker_count = self.marker_count + 1 if agrees else 1
                self.marker_pending = board
                self._trace(event='SC_observed_board_marker_candidate', sim_time=sim, board_base=board.tolist(), cameras=estimate['cameras'], mean_reprojection_px=estimate['mean_reprojection_px'], camera_disagreement_mm=estimate['camera_disagreement_mm'], camera_yaw_disagreement_deg=estimate['camera_yaw_disagreement_deg'], selected_port_prior_error_mm=estimate['selected_port_prior_error_mm'], inference_ms=estimate['inference_ms'], consistent_frames=self.marker_count, geometry_input=False)
                if self.marker_count >= 2:
                    self._replan_from_observed_board(board, state, sim)
            else:
                self.marker_count = 0
                self.marker_pending = None
        self.last_inspection = sim

    def _replan_from_observed_board(self, board, state, sim):
        original_board = np.asarray(self.params['nominal_board_base'])
        corrected_port = board @ np.linalg.inv(original_board) @ self.nominal_port
        self.nominal_port = corrected_port
        current = matrix_from_xyz_xyzw(state[:7])
        final = self._tcp_goal(corrected_port)
        start = np.linalg.inv(board) @ np.r_[current[:3, 3], 1.0]
        target = np.linalg.inv(board) @ np.r_[final[:3, 3], 1.0]
        outside = -0.23 if start[0] < -0.07 else 0.15
        beyond = 0.16
        low = self.params['bypass']['low_tcp_z_m']

        def point(x, y, z):
            v = board @ np.array([x, y, 0.0, 1.0])
            v[2] = z
            return v[:3]
        original = self.route_initial_rotation
        self.stages = [('clear_observed_card_row_side', point(outside, start[1], current[2, 3]), original), ('descend_outside_observed_card_row', point(outside, start[1], low), original), ('travel_beside_observed_card_row', point(outside, beyond, low), original), ('tilt_beyond_observed_card_row', point(outside, beyond, low), final[:3, :3]), ('sweep_toward_observed_port', point(target[0], beyond, low), final[:3, :3]), ('visual_acquisition', final[:3, 3], final[:3, :3])]
        self.index = 0
        self.stage_streak = 0
        self.reference = current.copy()
        self.compensation[:] = 0
        self.last_progress = sim
        self.best_error = float('inf')
        self.hold = None
        self.pose = None
        self.pending = None
        self.streak = 0
        self.pose_verified_time = -float('inf')
        self.marker_routed = True
        self._trace(event='SC_observed_board_route_selected', sim_time=sim, board_base=board.tolist(), corrected_requested_port_base=corrected_port.tolist(), start_board_local_m=start[:3].tolist(), outside_board_x_m=outside, selected_port_source='observed_marker_plus_static_asset_layout', true_scene_geometry_used=False, geometry_input=False)

    def predict_chunk(self, observation, task_vector, *, decision_time_s=None):
        tick = time.perf_counter()
        state = self.assembler.assemble_ros(observation)
        sim = self._control_sim_time(observation)
        current = matrix_from_xyz_xyzw(state[:7])
        c = self.spec['control']
        if self.first_sim_time is None:
            self.first_sim_time = sim
        self.last_elapsed_sim_sec = sim - self.first_sim_time
        if self.family is None:
            self._setup(task_vector, current, sim)
        inspected = sim - self.last_inspection >= c['inspection_interval_s']
        if inspected:
            self._inspect(observation, state, task_vector, sim)
        name, xyz, rotation = self.stages[self.index]
        marker_wait = self.family == 'sc' and os.environ.get('AIC_SC_TRANSPORT_ROUTE') == 'sc_marker_scout' and (name == 'look_for_board_marker') and (not self.marker_routed)
        if marker_wait:
            if self.marker_started is None:
                self.marker_started = sim
            if sim - self.marker_started > 6.0:
                self.rule_failure = 'observed_board_marker_not_found'
        goal = np.eye(4)
        goal[:3, 3] = xyz
        goal[:3, :3] = rotation
        acquisition = self.index == len(self.stages) - 1
        if acquisition:
            if self.acquisition_start is None:
                self.acquisition_start = sim
            verified_recent = self.pose is not None and sim - self.pose_verified_time <= 1.5
            if verified_recent:
                goal = self._tcp_goal(self.pose)
            elif sim - self.acquisition_start > 2.0:
                offsets = [(0, 0, 0), (0.01, 0, 0), (-0.01, 0, 0), (0, 0.01, 0), (0, -0.01, 0), (0, 0, 0.01), (0.02, 0, 0.01), (-0.02, 0, 0.01), (0, 0.02, 0.01), (0, -0.02, 0.01)]
                if self.scan_since is None:
                    self.scan_since = sim
                if sim - self.scan_since > 1.5:
                    self.scan_index = (self.scan_index + 1) % len(offsets)
                    self.scan_since = sim
                    self.reference = current.copy()
                    self.compensation[:] = 0
                goal[:3, 3] += np.asarray(offsets[self.scan_index])
            if sim - self.acquisition_start > c['acquisition_budget_s']:
                self.rule_failure = 'sample_visual_acquisition_budget'
        error = float(np.linalg.norm(goal[:3, 3] - state[:3]))
        angle = float((Rotation.from_matrix(current[:3, :3]).inv() * Rotation.from_matrix(goal[:3, :3])).magnitude())
        progress = error + 0.1 * angle
        if progress < self.best_error - 0.002:
            self.best_error = progress
            self.last_progress = sim
        close = error <= c['stage_tolerance_m'] and angle <= np.deg2rad(3.0)
        self.stage_streak = self.stage_streak + 1 if close else 0
        if not acquisition and self.stage_streak >= 3 and (not marker_wait):
            self._trace(event='rule_stage_completed', sim_time=sim, stage=name, position_error_mm=error * 1000)
            self.index += 1
            self.stage_streak = 0
            self.reference = current.copy()
            self.compensation[:] = 0
            self.best_error = float('inf')
            self.last_progress = sim
        if not acquisition and sim - self.last_progress > c['stage_no_progress_s']:
            self.rule_failure = 'sample_stage_no_measured_progress'
        stationary = np.linalg.norm(state[7:10]) <= 0.005 and np.linalg.norm(state[10:13]) <= np.deg2rad(3.0)
        fresh = bool(acquisition and self.pose is not None and (sim - self.pose_verified_time <= 1.5) and (np.linalg.norm(self._tcp_goal(self.pose)[:3, 3] - state[:3]) <= 0.015) and ((Rotation.from_matrix(self._tcp_goal(self.pose)[:3, :3]).inv() * Rotation.from_quat(state[3:7])).magnitude() <= np.deg2rad(3.0)))
        if acquisition and fresh and stationary:
            if self.hold is None:
                self.hold = sim
            self.rule_final_valid = sim - self.hold >= c['hold_s']
        else:
            self.hold = None
        if error < 0.03 and (not self.rule_failure):
            self.compensation += c['position_integral_gain'] * 0.2 * (goal[:3, 3] - state[:3])
            self.compensation = np.clip(self.compensation, -c['maximum_tracking_compensation_m'], c['maximum_tracking_compensation_m'])
        goal[:3, 3] += self.compensation
        if self.rule_failure:
            goal = current.copy()
            self.reference = current.copy()
        reference = self.reference
        distance = float(np.linalg.norm(goal[:3, 3] - reference[:3, 3]))
        rot = float((Rotation.from_matrix(reference[:3, :3]).inv() * Rotation.from_matrix(goal[:3, :3])).magnitude())
        duration = max(distance / c['speed_m_s'], rot / c['angular_speed_rad_s'], 0.2)
        interp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([reference[:3, :3], goal[:3, :3]])))
        targets = []
        for j in range(4):
            alpha = min((j + 1) / self.control_hz / duration, 1.0)
            target = np.eye(4)
            target[:3, 3] = reference[:3, 3] + alpha * (goal[:3, 3] - reference[:3, 3])
            target[:3, :3] = interp([alpha]).as_matrix()[0]
            relative = np.linalg.inv(current) @ target
            targets.append(np.r_[relative[:3, 3], relative[:3, :2].T.reshape(-1)])
        self.reference = target.copy()
        self._trace(event='sample_rule_progress', sim_time=sim, stage=name, error_mm=error * 1000, orientation_error_deg=float(np.rad2deg(angle)), failure=self.rule_failure, scan_index=self.scan_index, geometry_input=False)
        elapsed = (time.perf_counter() - tick) * 1000
        self._inference_latency_ms.append(elapsed)
        self._trace(event='sample_active_perception_timing', sim_time=sim, active=inspected, inference_ms=elapsed)
        return (np.asarray(targets, np.float32), np.zeros(9, np.float32), state)

    def _should_stop_transport(self, task_vector):
        return self.rule_final_valid or self.rule_failure is not None

    def _inspect_transport_handoff(self, observed, state, task_vector, get_observation, move_robot, decision_sim):
        if self.rule_final_valid and self.pose is not None:
            self._trace(event='rule_cached_visual_handoff', sim_time=decision_sim, report=dict(valid=True, reason='recent_consensus_standoff', requested_port=self.requested_port, requested_card=self.requested_card, predicted_opening_base_m=self.pose[:3, 3].tolist(), predicted_port_quat_xyzw=Rotation.from_matrix(self.pose[:3, :3]).as_quat().tolist(), visual_pose_age_s=decision_sim - self.pose_verified_time, runtime_true_geometry=False), geometry_input=False)
        self._trace(event='rule_terminal', sim_time=decision_sim, state=state.tolist(), acquired=self.rule_final_valid, failure=self.rule_failure, report=self.report, insertion_enabled=False)
        return True

class _RunSampleTransportRecovery(_RunSampleConservativeTransport):

    def __init__(self, parent_node):
        self._record_route = False
        self._route_history = []
        self._route_forces = []
        self._route_clear_checkpoint = None
        self._route_force_stop = False
        self._transport_attempt = 0
        self._resume_transport = False
        self._resume_pose = None
        self._resume_sim_time = None
        self._force_reference = None
        self._contact_monitor = TransportLoadStallMonitor()
        super().__init__(parent_node)

    def reset_for_task(self, task):
        if self._resume_transport:
            self.clear_pending_actions()
            self.assembler.reset()
            self.rule_failure = None
            self.rule_final_valid = False
            self.reference = self._resume_pose.copy()
            self.compensation[:] = 0
            self.best_error = float('inf')
            self.last_progress = self._resume_sim_time
            self.stage_streak = 0
            self.hold = None
            self._contact_monitor.reset()
            self._resume_transport = False
            return
        super().reset_for_task(task)

    def predict_chunk(self, observation, task_vector, *, decision_time_s=None):
        targets, estimated, state = super().predict_chunk(observation, task_vector, decision_time_s=decision_time_s)
        if self._record_route and self._force_reference is not None and (self.rule_failure is None) and (not self.rule_final_valid):
            sim = self._control_sim_time(observation)
            load = self._force_reference.measure(state[26:29], state[3:7])['load_change_norm_n']
            name, xyz, _ = self.stages[self.index]
            error = float(np.linalg.norm(np.asarray(xyz) - state[:3]))
            if self._contact_monitor.update(sim_time=sim, stage=name, goal_error_m=error, load_change_n=load):
                self._trace(event='transport_local_contact_detected', sim_time=sim, stage=name, load_change_n=load, goal_error_mm=error * 1000, reason='sustained_load_and_lost_measured_waypoint_progress', oracle_contact_used=False, geometry_input=False)
                self.clear_pending_actions()
                raise _TransportLocalContact()
        return (targets, estimated, state)

    def _calibrate_transport_force(self, get_observation, move_robot):
        self.assembler.reset()
        reference = InitialLoadReference(samples=10, minimum_quiet_span_s=0.35)
        last = -float('inf')
        started = time.monotonic()
        hold = None
        while time.monotonic() - started < 8:
            observation = get_observation()
            if observation is None:
                time.sleep(0.005)
                continue
            sim = self._control_sim_time(observation)
            if sim - last < 0.05:
                time.sleep(0.002)
                continue
            last = sim
            state = self.assembler.assemble_ros(observation)
            if hold is None:
                hold = matrix_from_xyz_xyzw(state[:7])
            self._command_local(move_robot, hold, owner='transport_force_free_hold', state=state, sim=sim)
            stamp = observation.wrist_wrench.header.stamp
            force_time = stamp.sec + stamp.nanosec * 1e-09
            if reference.last_stamp is not None and force_time <= reference.last_stamp:
                continue
            if abs(sim - force_time) > 0.1:
                continue
            if reference.add_initial_free_hold(stamp_s=force_time, force_sensor_n=state[26:29], tcp_quaternion_xyzw=state[3:7], linear_speed_mps=np.linalg.norm(state[7:10]), angular_speed_radps=np.linalg.norm(state[10:13])):
                self._force_reference = reference
                self._trace(event='transport_initial_load_reference_frozen', sim_time=sim, initial_reference_base_n=reference.reference.tolist(), quiet_samples=reference.samples, oracle_contact_used=False, geometry_input=False)
                return True
        self._trace(event='transport_local_recovery_unavailable', reason='no_quiet_initial_force_reference')
        return False

    def _command_local(self, move_robot, goal, *, owner, state, sim):
        if hasattr(self._parent_node, '_target_mode'):
            self._parent_node._target_mode = TargetMode.MODE_CARTESIAN
        stiffness, damping = self._cartesian_gains()
        self.set_pose_target(move_robot=move_robot, pose=build_pose_from_vectors(goal[:3, 3], Rotation.from_matrix(goal[:3, :3]).as_quat()), frame_id='base_link', stiffness=stiffness, damping=damping)
        self._trace(event='transport_local_recovery_command', sim_time=sim, owner=owner, state=state.tolist(), absolute_tcp_target=goal.tolist(), actor_owned=False, geometry_input=False)

    def _move_local(self, get_observation, move_robot, goal, *, owner, max_duration_s, maximum_load_change_n, clear_load_change_n=None, movement_axis=None, orthogonal_axis=None, minimum_projected_m=None):
        observation = get_observation()
        if observation is None:
            return None
        start_sim = self._control_sim_time(observation)
        state = self.assembler.assemble_ros(observation)
        start = matrix_from_xyz_xyzw(state[:7])
        last = -float('inf')
        started = time.monotonic()
        distance = float(np.linalg.norm(goal[:3, 3] - start[:3, 3]))
        stable = 0
        integral = np.zeros(3)
        while time.monotonic() - started < max_duration_s:
            observation = get_observation()
            if observation is None:
                time.sleep(0.005)
                continue
            sim = self._control_sim_time(observation)
            if sim - last < 0.05:
                time.sleep(0.002)
                continue
            dt = 0.05 if last < 0 else min(0.1, sim - last)
            last = sim
            state = self.assembler.assemble_ros(observation)
            measured = matrix_from_xyz_xyzw(state[:7])
            load = self._force_reference.measure(state[26:29], state[3:7])['load_change_norm_n']
            if load > maximum_load_change_n:
                self._trace(event='transport_local_recovery_abort', sim_time=sim, reason='load_growth', owner=owner, load_change_n=load, geometry_input=False)
                return None
            remaining = float(np.linalg.norm(goal[:3, 3] - measured[:3, 3]))
            clear = clear_load_change_n is None or load <= clear_load_change_n
            achieved = remaining <= 0.0015
            if minimum_projected_m is not None:
                displacement = measured[:3, 3] - start[:3, 3]
                projected = float(np.dot(displacement, movement_axis))
                orthogonal = float(np.dot(displacement, orthogonal_axis))
                achieved = projected >= minimum_projected_m and abs(orthogonal) <= 0.003
            stable = stable + 1 if achieved and clear and (np.linalg.norm(state[7:10]) < 0.008) else 0
            if stable >= 2:
                return (measured, state, sim)
            fraction = min(1.0, max(0.0, sim - start_sim) * 0.025 / max(distance, 1e-09))
            nominal = start[:3, 3] + fraction * (goal[:3, 3] - start[:3, 3])
            integral = np.clip(integral + 0.5 * dt * (nominal - state[:3]), -0.004, 0.004)
            command = start.copy()
            command[:3, 3] = nominal + integral
            self._command_local(move_robot, command, owner=owner, state=state, sim=sim)
        self._trace(event='transport_local_recovery_abort', reason='measured_arrival_timeout', owner=owner, geometry_input=False)
        return None

    def _recover_local(self, get_observation, move_robot, *, attempt):
        observation = get_observation()
        if observation is None:
            return False
        state = self.assembler.assemble_ros(observation)
        sim = self._control_sim_time(observation)
        current = matrix_from_xyz_xyzw(state[:7])
        load = self._force_reference.measure(state[26:29], state[3:7])['load_change_norm_n']
        if load > 45.0:
            self._trace(event='transport_local_recovery_abort', sim_time=sim, reason='too_much_initial_load', load_change_n=load, geometry_input=False)
            return False
        positions = [p[:3, 3] for p in self._route_history[-40:]] + [current[:3, 3]]
        try:
            axial = measured_approach_axis(positions)
        except ValueError:
            self._trace(event='transport_local_recovery_abort', sim_time=sim, reason='missing_measured_approach_direction', geometry_input=False)
            return False
        lateral = horizontal_lateral_axis(axial, 1 if attempt == 0 else -1)
        self._trace(event='transport_local_recovery_started', sim_time=sim, attempt=attempt + 1, load_change_n=load, approach_axis_base=axial.tolist(), lateral_axis_base=lateral.tolist(), retreat_limit_mm=12.0, oracle_contact_used=False, geometry_input=False)
        retreat = None
        for amount in (0.008, 0.012):
            goal = current.copy()
            goal[:3, 3] -= axial * amount
            retreat = self._move_local(get_observation, move_robot, goal, owner='transport_short_measured_retreat', max_duration_s=8.0, maximum_load_change_n=max(30.0, load + 8.0))
            if retreat is None:
                return False
            measured_state = retreat[1]
            remaining_load = self._force_reference.measure(measured_state[26:29], measured_state[3:7])['load_change_norm_n']
            if remaining_load <= 8.0:
                break
        if retreat is None:
            return False
        if remaining_load > 8.0:
            hold = retreat[0].copy()
            last = -float('inf')
            start_sim = retreat[2]
            settle_wall = time.monotonic()
            clear_streak = 0
            while time.monotonic() - settle_wall < 3.0:
                observation = get_observation()
                if observation is None:
                    time.sleep(0.005)
                    continue
                sim = self._control_sim_time(observation)
                if sim - last < 0.05:
                    time.sleep(0.002)
                    continue
                last = sim
                state = self.assembler.assemble_ros(observation)
                remaining_load = self._force_reference.measure(state[26:29], state[3:7])['load_change_norm_n']
                self._command_local(move_robot, hold, owner='transport_retreat_settle_hold', state=state, sim=sim)
                clear_streak = clear_streak + 1 if remaining_load <= 8.0 else 0
                if clear_streak >= 3:
                    retreat = (matrix_from_xyz_xyzw(state[:7]), state, sim)
                    break
                if sim - start_sim >= 1.0:
                    break
            if clear_streak < 3:
                self._trace(event='transport_local_recovery_abort', sim_time=last, reason='load_not_cleared_within_12mm', load_change_n=remaining_load, geometry_input=False)
                return False
        retreat_pose, retreat_state, retreat_sim = retreat
        measured_retreat = float(np.dot(current[:3, 3] - retreat_pose[:3, 3], axial))
        if measured_retreat < 0.005 or measured_retreat > 0.014:
            self._trace(event='transport_local_recovery_abort', sim_time=retreat_sim, reason='retreat_outside_measured_bounds', measured_retreat_mm=measured_retreat * 1000, geometry_input=False)
            return False
        amount = 0.008 if attempt == 0 else 0.012
        shifted = None
        for side in range(2):
            if side:
                lateral = -lateral
            goal = retreat_pose.copy()
            goal[:3, 3] += lateral * amount
            self._trace(event='transport_lateral_side_started', sim_time=retreat_sim, side=side + 1, axis_base=lateral.tolist(), requested_mm=amount * 1000, geometry_input=False)
            shifted = self._move_local(get_observation, move_robot, goal, owner='transport_lateral_retry', max_duration_s=9.0, maximum_load_change_n=10.0, clear_load_change_n=8.0, movement_axis=lateral, orthogonal_axis=axial, minimum_projected_m=0.004)
            if shifted is not None:
                break
            if side == 0:
                back = self._move_local(get_observation, move_robot, retreat_pose, owner='transport_lateral_unload', max_duration_s=5.0, maximum_load_change_n=30.0, clear_load_change_n=8.0)
                if back is None:
                    return False
        if shifted is None:
            return False
        final, state, sim = shifted
        displacement = final[:3, 3] - retreat_pose[:3, 3]
        lateral_m = float(np.dot(displacement, lateral))
        axial_drift = float(np.dot(displacement, axial))
        if lateral_m < 0.004 or abs(axial_drift) > 0.003:
            self._trace(event='transport_local_recovery_abort', sim_time=sim, reason='lateral_not_measured_or_axial_drift', measured_lateral_mm=lateral_m * 1000, axial_drift_mm=axial_drift * 1000, geometry_input=False)
            return False
        offset = lateral * lateral_m
        for index in range(self.index, len(self.stages) - 1):
            name, xyz, rotation = self.stages[index]
            self.stages[index] = (name, np.asarray(xyz) + offset, rotation)
        self._resume_pose = final.copy()
        self._resume_sim_time = sim
        self._route_history = [final.copy()]
        self._route_forces = [float(np.linalg.norm(state[26:29]))]
        self._route_force_stop = False
        self._trace(event='transport_local_recovery_complete', sim_time=sim, attempt=attempt + 1, measured_retreat_mm=measured_retreat * 1000, measured_lateral_mm=lateral_m * 1000, axial_drift_mm=axial_drift * 1000, shifted_transport_stage_count=max(0, len(self.stages) - self.index - 1), resumed_stage=self.stages[self.index][0], actor_owned=False, geometry_input=False)
        return True

    def _trace(self, **record):
        record['transport_attempt'] = self._transport_attempt
        if self._record_route and record.get('event') in ('command', 'terminal_before_return') and ('state' in record):
            state = np.asarray(record['state'], float)
            pose = matrix_from_xyz_xyzw(state[:7])
            if not self._route_history or np.linalg.norm(pose[:3, 3] - self._route_history[-1][:3, 3]) >= 0.002 or (Rotation.from_matrix(pose[:3, :3]).inv() * Rotation.from_matrix(self._route_history[-1][:3, :3])).magnitude() >= np.deg2rad(2):
                self._route_history.append(pose)
                self._route_forces.append(float(np.linalg.norm(state[26:29])))
            if record.get('reason') == 'force_stop':
                self._route_force_stop = True
        if self._record_route and record.get('event') == 'rule_stage_completed' and (record.get('stage') == 'clear_start_side') and self._route_history:
            self._route_clear_checkpoint = len(self._route_history) - 1
        super()._trace(**record)

    def insert_cable(self, task, get_observation, move_robot, send_feedback, **kwargs):
        observation = get_observation()
        if observation is None:
            return False
        start = self._control_sim_time(observation)
        original = copy.deepcopy(self.spec['families']['sc']['bypass'])
        wall_start = time.monotonic()
        retained = [observation]

        def bounded_observation():
            value = get_observation()
            if value is not None:
                stamp = self._control_sim_time(value)
                if stamp < start or stamp - start >= 155 or time.monotonic() - wall_start >= 290:
                    raise _TransportRecoveryBudget('whole_transport_budget')
                retained[0] = value
            return value
        try:
            self._transport_attempt = 0
            self._route_history = []
            self._route_forces = []
            self._route_clear_checkpoint = None
            self._route_force_stop = False
            self._record_route = False
            self._resume_transport = False
            self._force_reference = None
            self._contact_monitor.reset()
            if not self._calibrate_transport_force(bounded_observation, move_robot):
                return False
            for attempt in range(3):
                self._transport_attempt = attempt
                self._route_force_stop = False
                self._record_route = True
                contact = False
                try:
                    super().insert_cable(task, bounded_observation, move_robot, send_feedback, **kwargs)
                except _TransportLocalContact:
                    contact = True
                self._record_route = False
                if self.rule_final_valid:
                    return False
                if not (contact or self._route_force_stop):
                    return False
                if attempt >= 2:
                    self._trace(event='transport_local_recovery_abort', reason='two_retry_budget_exhausted', geometry_input=False)
                    return False
                if not self._recover_local(bounded_observation, move_robot, attempt=attempt):
                    return False
                self._resume_transport = True
        except _TransportRecoveryBudget:
            self.rule_final_valid = False
            self._trace(event='transport_recovery_budget_exhausted', geometry_input=False)
            return False
        finally:
            self._record_route = False
            self._resume_transport = False
            self.spec['families']['sc']['bypass'] = original
            current = retained[0]
            if current is not None:
                self._trace(event='terminal_before_return', reason='sample_transport_recovery_complete', sim_time=self._control_sim_time(current), state=self.assembler.assemble_ros(current).tolist(), observation_source='current_outer_action_before_return', geometry_input=False)
        return False

class _RunObservedEmptyBoardTransport(_RunSampleTransportRecovery):

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.empty_row = EmptyRowObservation(os.environ['AIC_EMPTY_ROW_CHECKPOINT'], os.environ['AIC_EMPTY_ROW_APPROVAL'], str(self.device))

    def _reset(self):
        super()._reset()
        self.empty_row_streak = 0
        self.empty_row_last_stamps = None
        self.empty_shortcut_selected = False

class _RunObservedEmptyBoardWindowV40(_RunObservedEmptyBoardTransport):

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.pending_initial_receipt = None

    def _inspect(self, observation, state, task, sim):
        return _RunSampleConservativeTransport._inspect(self, observation, state, task, sim)

    def _setup(self, task, current, sim):
        super()._setup(task, current, sim)
        receipt = self.pending_initial_receipt
        self.pending_initial_receipt = None
        if self.family != 'sc' or self._transport_attempt != 0 or (not receipt_matches(receipt, task, sim, current)):
            return
        port = self.nominal_port.copy()
        goal = self._tcp_goal(port)
        height = max(float(current[2, 3]), float(goal[2, 3] + 0.04))
        if height > 0.55 or np.linalg.norm(goal[:3, 3] - current[:3, 3]) > 0.6:
            return
        xyz = current[:3, 3]
        rotation = self.reference[:3, :3].copy()
        self.stages = [('empty_row_rise_clearance', np.array([xyz[0], xyz[1], height]), rotation), ('empty_row_transfer_clearance', np.array([goal[0, 3], goal[1, 3], height]), rotation), ('empty_row_orient_above_target', np.array([goal[0, 3], goal[1, 3], height]), goal[:3, :3]), ('empty_row_visual_acquisition', goal[:3, 3], goal[:3, :3])]
        self.empty_shortcut_selected = True
        self._trace(event='empty_row_shortcut_selected', sim_time=sim, source='three fresh stationary pre-route image decisions', port_waypoint_source='public_training_prior_only', final_requested_visual_gate_unchanged=True, reference_preserved=True, receipt=receipt, geometry_input=False)

class _RunObservedShortSFPTransportV43(_RunObservedEmptyBoardWindowV40):

    def _tcp_goal(self, port):
        goal = super()._tcp_goal(port)
        if self.family == 'sfp':
            goal[:3, 3] += port[:3, 2] * 0.03
        return goal

class _RunObservedDirectSFPTransportV44(_RunObservedShortSFPTransportV43):

    def _tcp_goal(self, port):
        goal = super()._tcp_goal(port)
        if self.family == 'sfp':
            goal[:3, 3] += port[:3, 2] * 0.04
        return goal

    def _build_stages(self, current, port):
        if self.family != 'sfp':
            return super()._build_stages(current, port)
        final = self._tcp_goal(port)
        height = max(float(current[2, 3]), float(final[2, 3] + 0.01))
        above_start = current[:3, 3].copy()
        above_start[2] = height
        above_target = final[:3, 3].copy()
        above_target[2] = height
        self.stages = [('SFP_direct_clearance', above_start, current[:3, :3].copy()), ('SFP_transfer_and_orient_clear', above_target, final[:3, :3].copy()), ('SFP_direct_visual_acquisition', final[:3, 3], final[:3, :3])]

class _RunObservedEarlyVisualAcquireV45(_RunObservedDirectSFPTransportV44):

    def _reset(self):
        super()._reset()
        self.early_active = False
        self.early_reference = None
        self.early_stamps = None
        self.early_fresh_count = 0
        self.early_rejections = 0
        self.initial_public_reference = None
        self.early_causal_poses = []

    def _tcp_goal(self, port):
        if self.early_active:
            return self.early_reference.copy()
        return super()._tcp_goal(port)

    def _build_stages(self, current, port):
        super()._build_stages(current, port)
        if self.family == 'sfp':
            self.fallback_stages = self.stages
            self.early_reference = self.initial_public_reference.copy() if self.initial_public_reference is not None else current.copy()
            self.early_active = True
            self.early_started = self.last_progress
            hold = self.early_reference
            self.stages = [('SFP_initial_visual_acquisition', hold[:3, 3].copy(), hold[:3, :3].copy())]

    def _setup(self, task, current, sim):
        super()._setup(task, current, sim)
        if self.early_active:
            self.reference = self.early_reference.copy()
        if self.family == 'sc' and self.empty_shortcut_selected:
            rise, transfer, orient, acquire = self.stages
            self.stages = [rise, (transfer[0], transfer[1], orient[2]), acquire]
            self._trace(event='empty_row_transfer_orientation_combined', sim_time=sim, source='same_three_fresh_stationary_scene_decisions', occupied_scene_route_unchanged=True, geometry_input=False)

    def _inspect(self, observation, state, task, sim):
        super()._inspect(observation, state, task, sim)
        if not self.early_active:
            return
        stamps = tuple((getattr(observation, c + '_image').header.stamp.sec * 1000000000 + getattr(observation, c + '_image').header.stamp.nanosec for c in ('center', 'left', 'right')))
        distinct = self.early_stamps is None or all((a > b for a, b in zip(stamps, self.early_stamps)))
        ages = sim - np.asarray(stamps) * 1e-09
        fresh = distinct and ages.min() >= -0.001 and (ages.max() <= 0.1) and (max(stamps) - min(stamps) <= 50000000)
        packet_timing = None
        packet_mode = getattr(self, 'use_packet_camera_timing', False)
        if packet_mode:
            from .runtime.public_packet_camera_timing import camera_receipt
            packet_timing = camera_receipt(sim, stamps, self.early_stamps)
            fresh = bool(packet_timing['accepted'] and np.linalg.norm(state[7:10]) <= 0.005 and (np.linalg.norm(state[10:13]) <= 0.03))
        if not packet_mode or fresh:
            self.early_stamps = stamps
        native = self.report['native_pose_component']
        plausible = False
        if native['accepted']:
            port = np.asarray(native['base_port_matrix'])
            tcp = np.linalg.inv(port) @ matrix_from_xyz_xyzw(state[:7])
            nominal_tip = tcp[:3, 3] + tcp[:3, :3] @ np.asarray(self.params['body_tip_offset_m'])
            plausible = bool(0.12 <= -tcp[2, 3] <= 0.25 and 0.07 <= -nominal_tip[2] <= 0.15 and (np.linalg.norm(nominal_tip[:2]) <= 0.05))
        if packet_mode:
            if fresh:
                if plausible:
                    self.early_causal_poses = (self.early_causal_poses + [port.copy()])[-3:]
                    shift = max((np.linalg.norm(a[:3, 3] - b[:3, 3]) for a in self.early_causal_poses for b in self.early_causal_poses))
                    angle = max((np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1.0, 1.0)) for a in self.early_causal_poses for b in self.early_causal_poses))
                    if shift > 0.001 or angle > np.deg2rad(1.0):
                        self.early_causal_poses = [port.copy()]
                    self.early_rejections = 0
                else:
                    self.early_causal_poses = []
                    self.early_rejections += 1
                self.early_fresh_count = len(self.early_causal_poses)
        else:
            self.early_fresh_count = self.early_fresh_count + 1 if fresh and plausible else 0
            if fresh:
                self.early_rejections = self.early_rejections + 1 if not plausible else 0
        self.report['valid'] = bool(self.report['valid'] and fresh and (self.early_fresh_count >= 3))
        self.report['reason'] = 'fresh_requested_port_in_initial_approach_region' if self.report['valid'] else 'initial_acquisition_requires_fresh_views_and_public_region'
        self._trace(event='SFP_initial_acquisition_check', sim_time=sim, valid=self.report['valid'], plausible_public_nominal_tip=plausible, fresh_consistent_observations=self.early_fresh_count, image_ages_s=ages.tolist(), packet_timing=packet_timing, geometry_input=False)

    def predict_chunk(self, observation, task_vector, *, decision_time_s=None):
        sim = self._control_sim_time(observation)
        if self.family is None:
            value = observation.controller_state.reference_tcp_pose
            p, q = (value.position, value.orientation)
            self.initial_public_reference = matrix_from_xyz_xyzw([p.x, p.y, p.z, q.x, q.y, q.z, q.w])
        reject_limit = getattr(self, 'initial_reject_limit', 0)
        repeated_rejection = reject_limit > 0 and self.early_rejections >= reject_limit
        if self.early_active and (sim - self.early_started > 3.0 or repeated_rejection) and (not self.rule_final_valid):
            state = self.assembler.assemble_ros(observation)
            self.early_active = False
            self.stages = self.fallback_stages
            self.index = 0
            self.stage_streak = 0
            self.reference = matrix_from_xyz_xyzw(state[:7])
            self.compensation[:] = 0.0
            self.hold = None
            self.acquisition_start = None
            self.best_error = float('inf')
            self.last_progress = sim
            self._trace(event='SFP_initial_acquisition_route_fallback', sim_time=sim, reason='repeated_fresh_invalid_initial_views' if repeated_rejection else 'no_three_fresh_valid_requested_views_within3s', geometry_input=False)
        return super().predict_chunk(observation, task_vector, decision_time_s=decision_time_s)

class _RunObservedEarlyVisualAcquireV46(_RunObservedEarlyVisualAcquireV45):
    initial_reject_limit = 3

    def _reset(self):
        super()._reset()
        self.coarse_candidates = []
        self.coarse_stable = None
        self.coarse_fallback_applied = False

    def _fallback_from_port(self, current, port):
        final = _RunObservedDirectSFPTransportV44._tcp_goal(self, port)
        above = final[:3, 3].copy()
        above[2] = max(float(current[2, 3]), float(above[2]))
        lift = float(os.environ.get('AIC_SFP_FALLBACK_LIFT_M', '0'))
        if not 0.0 <= lift <= 0.1:
            raise ValueError('SFP fallback lift must be in [0, 0.10] m')
        before = []
        if lift:
            lifted = current[:3, 3].copy()
            lifted[2] += lift
            above[2] = max(above[2], lifted[2])
            before = [('SFP_lift_before_transfer', lifted, current[:3, :3])]
        return [*before, ('SFP_transfer_orientation_without_blind_descent', above, final[:3, :3]), ('SFP_requested_native_visual_acquisition', final[:3, 3], final[:3, :3])]

    def _build_stages(self, current, port):
        super()._build_stages(current, port)
        if self.family != 'sfp':
            return
        self.fallback_stages = self._fallback_from_port(current, port)

    def _inspect(self, observation, state, task, sim):
        super()._inspect(observation, state, task, sim)
        if self.family != 'sfp' or not self.early_active:
            return
        result = self.report['native_pose_component']
        pixels = result.get('coarse_pixels')
        if pixels is None or min(result.get('projected_inframe_probability', [0])) < 0.95:
            self.coarse_candidates = []
            return
        names = ('center', 'left', 'right')
        cameras = {c: matrix_from_xyz_xyzw(state[:7]) @ self.component.optics[c] for c in names}
        try:
            point = np.asarray(triangulate(dict(zip(names, pixels)), cameras), float)
            residual = []
            for c, pixel in zip(names, pixels):
                xyz = np.linalg.inv(cameras[c]) @ np.r_[point, 1.0]
                projected = xyz[:2] / xyz[2] * FOCAL + [576.0, 512.0]
                residual.append(float(np.linalg.norm(projected - np.asarray(pixel))))
            nominal = self.nominal_port[:3, 3]
            plausible = np.isfinite(point).all() and max(residual) <= 14.0 and (abs(point[0] - nominal[0]) <= 0.08) and (abs(point[1] - nominal[1]) <= 0.35) and (abs(point[2] - nominal[2]) <= 0.05)
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
            plausible = False
            residual = []
            point = None
        if not plausible:
            self.coarse_candidates = []
            return
        if self.coarse_candidates and (sim - self.coarse_candidates[-1][0] > 0.8 or np.linalg.norm(point - self.coarse_candidates[-1][1]) > 0.015):
            self.coarse_candidates = []
        self.coarse_candidates = (self.coarse_candidates + [(sim, point)])[-3:]
        if len(self.coarse_candidates) >= 2:
            self.coarse_stable = np.mean([x[1] for x in self.coarse_candidates], axis=0)
            self._trace(event='SFP_consistent_coarse_waypoint', sim_time=sim, predicted_opening_base_m=self.coarse_stable.tolist(), reprojection_px=residual, observations=len(self.coarse_candidates), role='transport_waypoint_only', geometry_input=False)

    def predict_chunk(self, observation, task_vector, *, decision_time_s=None):
        if self.family == 'sfp' and self.early_active:
            self.initial_reject_limit = 3
        if self.early_active and self.coarse_stable is not None and (not self.coarse_fallback_applied):
            state = self.assembler.assemble_ros(observation)
            port = self.nominal_port.copy()
            port[:3, 3] = self.coarse_stable
            self.fallback_stages = self._fallback_from_port(matrix_from_xyz_xyzw(state[:7]), port)
            self.coarse_fallback_applied = True
            self._trace(event='SFP_coarse_guided_safe_fallback', predicted_opening_base_m=self.coarse_stable.tolist(), true_port_geometry_used=False, geometry_input=False)
        return super().predict_chunk(observation, task_vector, decision_time_s=decision_time_s)

class _RunObservedDirectEmptySCV49(_RunObservedEarlyVisualAcquireV46):

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.base_control = dict(self.spec['control'])

    def _reset(self):
        super()._reset()
        self.direct_empty_selected = False
        if hasattr(self, 'base_control'):
            self.spec['control'].clear()
            self.spec['control'].update(self.base_control)

    def _tcp_goal(self, port):
        goal = super()._tcp_goal(port)
        if self.family == 'sc' and self.direct_empty_selected:
            goal[:3, 3] += port[:3, 2] * 0.03
        return goal

    def _setup(self, task, current, sim):
        super()._setup(task, current, sim)
        if self.family != 'sc' or not self.empty_shortcut_selected:
            return
        candidate = super()._tcp_goal(self.nominal_port)
        candidate[:3, 3] += self.nominal_port[:3, 2] * 0.03
        if min(float(current[2, 3]), float(candidate[2, 3])) < 0.14 or np.linalg.norm(candidate[:3, 3] - current[:3, 3]) > 0.6:
            return
        self.direct_empty_selected = True
        self.spec['control'].update(speed_m_s=0.12, angular_speed_rad_s=0.8)
        self.stages = [('SC_empty_diagonal_visual_acquisition', candidate[:3, 3], candidate[:3, :3])]
        self._trace(event='SC_empty_diagonal_acquisition_selected', sim_time=sim, source='three_fresh_stationary_full_RGB_empty_row_decisions', nominal_tip_standoff_mm=120, minimum_nominal_TCP_height_m=0.14, occupied_or_uncertain_route_unchanged=True, native_requested_acquisition_still_required=True, geometry_input=False)

class _RunPredictedPortLocal(_RunSharedPortTCP):

    @staticmethod
    def raw_rgb(message):
        enc = message.encoding.lower()
        count = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4}.get(enc)
        if count is None:
            raise ValueError(enc)
        raw = np.frombuffer(message.data, np.uint8).reshape(message.height, message.step)
        rgb = raw[:, :message.width * count].reshape(message.height, message.width, count)[..., :3]
        return np.ascontiguousarray(rgb[..., ::-1] if enc.startswith('bgr') else rgb)

class _RunRuleTransport(_RunSharedPortTCP):

    @staticmethod
    def native(message):
        channels = 4 if message.encoding.lower() in ('rgba8', 'bgra8') else 3
        raw = np.frombuffer(message.data, np.uint8).reshape(message.height, message.step)
        rgb = raw[:, :message.width * channels].reshape(message.height, message.width, channels)[..., :3]
        return rgb[..., ::-1].copy() if message.encoding.lower().startswith('bgr') else rgb.copy()

class _RunVisualConnectorServoDiagnosticV2(_RunContinuousConnectorInsertionV2):

    def _execute(self, task, get_observation, move_robot, send_feedback, **kwargs):
        original = self.local.insert_cable
        self.local.insert_cable = self._visual_servo_insert
        try:
            return super()._execute(task, get_observation, move_robot, send_feedback, **kwargs)
        finally:
            self.local.insert_cable = original

class _RunVisualConnectorServoDiagnosticV3(_RunVisualConnectorServoDiagnosticV2):

    def _warm_grasp(self):
        started = time.perf_counter()
        super()._warm_grasp()
        with torch.inference_mode():
            for family, name in [('sc', 'sc_orientation_jitter_v2'), ('sfp', 'sfp_direct_pose')]:
                if family not in self.connector_models:
                    self.connector_models[family] = NativeConnectorPoseObservation(ROOT / name / 'best.pt', str(self.local.device))
                self.connector_models[family].model(torch.zeros(1, 3, 3, 192, 192, dtype=torch.uint8, device=self.local.device))
        if self.local.device.type == 'cuda':
            torch.cuda.synchronize(self.local.device)
        self._trace(event='visual_servo_pose_networks_warmed', initialization_ms=(time.perf_counter() - started) * 1000, synthetic_pixels_bound_as_pose=False, geometry_input=False)

class _RunVisualConnectorServoDiagnosticV4(_RunVisualConnectorServoDiagnosticV3):
    pass

class _RunVisualConnectorServoRecoveryV5(_RunVisualConnectorServoDiagnosticV4):

    def __init__(self, parent_node):
        requested = os.environ.get('AIC_SHARED_RECOVERY_MODE', 'none')
        if requested not in ('none', 'local', 'hierarchical'):
            raise ValueError('Unknown recovery arm')
        previous = os.environ.get('AIC_SHARED_RECOVERY_MODE')
        os.environ['AIC_SHARED_RECOVERY_MODE'] = 'none'
        try:
            super().__init__(parent_node)
        finally:
            if previous is None:
                os.environ.pop('AIC_SHARED_RECOVERY_MODE', None)
            else:
                os.environ['AIC_SHARED_RECOVERY_MODE'] = previous
        self.requested_recovery = requested
        self.mode = requested
        self.servo_goal_rotation = np.eye(3)
        self.angle_retry = 0
        self.recovery_seeded_supervisor = None
        self.approach_measured_history = []
        self.recovery_command = None
        self.recovery_command_stamp = None
        self.servo_stall_history = []

    def _execute(self, task, get_observation, move_robot, send_feedback, **kwargs):
        self.mode = self.requested_recovery if self.local.task_vector(task)[1] == 1 else 'none'
        self.angle_retry = 0
        self.servo_goal_rotation = np.eye(3)
        self.servo_stall_history = []
        return super()._execute(task, get_observation, move_robot, send_feedback, **kwargs)

    def _lock(self, task_vector, get_observation, move_robot, *, fresh_context):
        history = []

        def observed():
            value = get_observation()
            if value is not None:
                stamp = self.local._control_sim_time(value)
                if not history or stamp > history[-1][0]:
                    history.append((stamp, self.local.assembler.assemble_ros(value).copy()))
            return value
        accepted = super()._lock(task_vector, observed, move_robot, fresh_context=fresh_context)
        if accepted:
            self.approach_measured_history = history
        return accepted

    def _supervise(self, state, absolute):
        if self.recovery_seeded_supervisor is not self.supervisor:
            for stamp, measured in self.approach_measured_history:
                self.supervisor._append_history(measured_pose(measured), self._force(measured), 'approach', self.route)
            self.recovery_seeded_supervisor = self.supervisor
            self._trace(event='backoff_approach_history_seeded', observed_samples=len(self.approach_measured_history), geometry_input=False)
        return super()._supervise(state, absolute)

    def _command(self, move_robot, matrix, *, owner):
        super()._command(move_robot, matrix, owner=owner)
        if owner in ('measured_path_backtrack', 'supervisor_lateral_exploration'):
            self.recovery_command = np.asarray(matrix).copy()
            self.recovery_command_stamp = self.local.last_observation_stamp

class _RunVisualConnectorServoRecoveryV6(_RunVisualConnectorServoRecoveryV5):
    pass

class _RunVisualConnectorServoRecoveryV7(_RunVisualConnectorServoRecoveryV6):

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.unloaded_reference_bias_base_m = np.zeros(3)

    def _lock(self, task_vector, get_observation, move_robot, *, fresh_context):
        accepted = super()._lock(task_vector, get_observation, move_robot, fresh_context=fresh_context)
        if accepted:
            end = self.approach_measured_history[-1][0]
            free = [state[13:16] for stamp, state in self.approach_measured_history if end - stamp <= 0.5 and self._force(state) < 3.0 and (np.linalg.norm(state[7:10]) < 0.005)]
            if not free:
                raise ValueError('No unloaded causal tracking-error samples for measured recovery')
            bias = np.median(free, axis=0)
            if not np.isfinite(bias).all() or np.linalg.norm(bias) > 0.02:
                raise ValueError('Invalid free-space tracking compensation')
            self.unloaded_reference_bias_base_m = bias
            self._trace(event='free_tracking_bias_retained', sim_time=end, bias_base_m=bias.tolist(), source='public controller tcp_error at free25mm approach; same controller gains', geometry_input=False)
        return accepted

class _RunVisualConnectorServoRecoveryV8(_RunVisualConnectorServoRecoveryV7):

    def _supervise_servo_stall(self, state, issued, point, stamp, force, depth):
        self.servo_stall_history.append((stamp, float(point[2, 3])))
        self.servo_stall_history = [x for x in self.servo_stall_history if stamp - x[0] <= 0.25]
        if len(self.servo_stall_history) < 2:
            return
        elapsed = stamp - self.servo_stall_history[0][0]
        gain = point[2, 3] - self.servo_stall_history[0][1]
        excess = issued[:3, 3] - state[:3] - self.unloaded_reference_bias_base_m
        axial_pressure = float(self.local.visual_port[:3, 2] @ excess)
        if elapsed >= 0.19 and gain <= 0.00015 and (axial_pressure >= 0.001) and (force >= 1.5) and (point[2, 3] < depth - 0.0005):
            phase = 'contact' if point[2, 3] > -0.005 else 'alignment'
            self.supervisor._trigger(measured_pose(state), force, phase, self.route)
            decision = self.supervisor.step(pose=measured_pose(state), force_n=force, prior_command_m=self.last_command_m, tracking_error_m=float(np.linalg.norm(issued[:3, 3] - state[:3])), phase=phase, route=self.route)
            self._trace(event='loaded_servo_stall_recovery', sim_time=stamp, load_change_n=force, progress_window_s=elapsed, axial_gain_mm=float(gain * 1000), excess_axial_tracking_pressure_mm=axial_pressure * 1000, reason='early_actual_load_and_excess_forward_tracking_pressure', synthetic_force_used=False, geometry_input=False, level=decision.level)
            self.local.clear_pending_actions()
            raise _RecoveryTransfer(decision)

class _RunVisualConnectorServoRecoveryV10(_RunVisualConnectorServoRecoveryV8):

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.recovery_tracking_integral = np.zeros(3)
        self.recovery_tracking_stamp = None

    def _retreat(self, transfer, get_observation, move_robot):
        self.recovery_tracking_integral = np.zeros(3)
        self.recovery_tracking_stamp = None
        return super()._retreat(transfer, get_observation, move_robot)

    def _command(self, move_robot, matrix, *, owner):
        if owner in ('measured_path_backtrack', 'supervisor_lateral_exploration'):
            nominal = np.asarray(matrix).copy()
            matrix = nominal.copy()
            stamp = self.local.last_observation_stamp
            state = self.retained_terminal['state'] if self.retained_terminal is not None else None
            if state is None:
                raise ValueError('Recovery reference needs the retained public measured state')
            dt = 0.0 if self.recovery_tracking_stamp is None else min(0.1, max(0.0, stamp - self.recovery_tracking_stamp))
            self.recovery_tracking_stamp = stamp
            error = nominal[:3, 3] - np.asarray(state[:3])
            self.recovery_tracking_integral = np.clip(self.recovery_tracking_integral + 0.6 * dt * error, -0.005, 0.005)
            matrix[:3, 3] += self.unloaded_reference_bias_base_m + self.recovery_tracking_integral
            self._trace(event='recovery_waypoint_reference_compensated', sim_time=stamp, owner=owner, measured_waypoint=nominal.tolist(), issued_reference=matrix.tolist(), bias_base_m=self.unloaded_reference_bias_base_m.tolist(), integral_base_m=self.recovery_tracking_integral.tolist(), measured_state=state, geometry_input=False)
        _RunVisualConnectorServoRecoveryV5._command(self, move_robot, matrix, owner=owner)

class _RunVisualConnectorServoRecoveryV12(_RunVisualConnectorServoRecoveryV10):

    def _lateral_retry(self, get_observation, move_robot, *, route_level):
        observation = get_observation()
        if observation is None:
            return False
        state = self.local.assembler.assemble_ros(observation)
        start = matrix_from_xyz_xyzw(state[:7])
        inverse = np.linalg.inv(self.local.visual_port)
        body = self.local.visual_body_tip
        tip = (inverse @ np.r_[start[:3, 3] + start[:3, :3] @ body, 1.0])[:3]
        if tip[2] > -0.005:
            self._trace(event='recovery_abort', reason='lateral_retry_while_plug_may_be_inside')
            return False
        retreat = start[:3, 3] - np.asarray(self.blocked_pose.xyz)
        direction = lateral_retry_direction(retreat, self.local.visual_port[:3, 2], self.exploration_rng)
        amplitude = 0.01 if route_level else 0.003
        while amplitude >= 0.0005:
            candidate = tip + inverse[:3, :3] @ (direction * amplitude)
            if np.linalg.norm(candidate[:2]) <= 0.025 and candidate[2] <= -0.005:
                break
            amplitude *= 0.5
        if amplitude < 0.0005 or (route_level and amplitude < 0.003):
            return False
        goal = start[:3, 3] + direction * amplitude
        started = time.monotonic()
        first = None
        last = None
        arrival = 0
        integral = np.zeros(3)
        self.local.clear_pending_actions()
        self.recovery_command = None
        self._trace(event='lateral_retry_started', sampled_direction_base=direction.tolist(), amplitude_m=amplitude, fixed_measured_goal_base_m=goal.tolist(), integral_reset=True, geometry_input=False)
        while time.monotonic() - started < 15 and time.monotonic() < self.episode_deadline - 2:
            observation = get_observation()
            if observation is None:
                time.sleep(0.005)
                continue
            sim = self.local._control_sim_time(observation)
            if sim - self.insertion_start_sim > self.total_budget_s:
                return False
            if last is not None and sim - last < 0.049:
                time.sleep(0.002)
                continue
            dt = 0.05 if last is None else min(0.1, sim - last)
            last = sim
            state = self.local.assembler.assemble_ros(observation)
            current = matrix_from_xyz_xyzw(state[:7])
            force = self._force(state)
            if first is None:
                first = sim
            if force >= 8 or np.linalg.norm(state[26:29]) > 40:
                self._trace(event='recovery_abort', reason='new_load_during_lateral_retry', load_change_n=force)
                return False
            self.supervisor._append_history(measured_pose(state), force, 'supervisor_lateral_retry', self.route)
            travelled = min(amplitude, 0.005 * (sim - first))
            reference = start[:3, 3] + direction * travelled
            integral = np.clip(integral + 0.6 * dt * (reference - state[:3]), -0.005, 0.005)
            target = start.copy()
            target[:3, 3] = reference + self.unloaded_reference_bias_base_m + integral
            _RunVisualConnectorServoRecoveryV5._command(self, move_robot, target, owner='supervisor_lateral_exploration')
            distance = float(np.linalg.norm(goal - state[:3]))
            self._trace(event='lateral_retry_tracking', sim_time=sim, fixed_goal_base_m=goal.tolist(), nominal_reference_base_m=reference.tolist(), integral_base_m=integral.tolist(), measured_state=state.tolist(), remaining_mm=distance * 1000, load_change_n=force, geometry_input=False)
            arrival = arrival + 1 if travelled >= amplitude and distance < 0.00075 and (np.linalg.norm(state[7:10]) < 0.005) else 0
            if arrival >= 3:
                moved = float(np.linalg.norm(state[:3] - start[:3, 3]))
                if route_level and moved < 0.003:
                    return False
                self.last_measured_lateral_retry_m = moved
                self._trace(event='lateral_retry_complete', sim_time=sim, route_level=route_level, measured_displacement_m=(state[:3] - start[:3, 3]).tolist(), sampled_direction_base=direction.tolist(), amplitude_m=amplitude, seed=self.exploration_seed, actor_owned=False, exploration_source='supervisor_coherent_direction_not_RL', geometry_input=False)
                if self.recovery_command is None:
                    raise ValueError('Missing actual issued lateral reference')
                self.local.clear_pending_actions()
                self.local.governor = None
                self.local.governor_seed = (self.recovery_command.copy(), self.recovery_command_stamp)
                self.servo_stall_history = []
                self.angle_retry += 1
                proposals = [(2.5, 0, 0), (-2.5, 0, 0), (0, 2.5, 0), (0, -2.5, 0), (0, 0, 2.5), (0, 0, -2.5)]
                if self.angle_retry > len(proposals):
                    self._trace(event='recovery_abort', reason='bounded_angular_retry_budget_exhausted')
                    return False
                self.servo_goal_rotation = Rotation.from_rotvec(np.deg2rad(proposals[self.angle_retry - 1])).as_matrix()
                self._trace(event='backoff_reference_reseeded', sim_time=self.recovery_command_stamp, issued_tcp_target=self.recovery_command.tolist(), goal_rotation_offset_rotvec_deg=proposals[self.angle_retry - 1], exploration_source='fixed_symmetric_angles_after_observed_clearance_not_RL', geometry_input=False)
                return True
        self._trace(event='recovery_abort', reason='lateral_retry_timeout', sim_time=last)
        return False

class _RunVisualConnectorServoRecoveryV13(_RunVisualConnectorServoRecoveryV12):

    def _supervise_servo_stall(self, state, issued, point, stamp, force, depth):
        if point[2, 3] < -0.005:
            self.servo_stall_history = []
            return
        return super()._supervise_servo_stall(state, issued, point, stamp, force, depth)

class _RunVisualConnectorServoRecoveryV21(_RunVisualConnectorServoRecoveryV13):

    def __init__(self, parent_node):
        super().__init__(parent_node)
        approval = json.loads(Path(os.environ['AIC_SC_CLOSE_GRASP_APPROVAL']).read_text())
        experimental = approval.get('prediction_kind') == 'experimental_composite_pose'
        if experimental and os.environ.get('AIC_SC_CLOSE_DIAGNOSTIC_TRIAL') != '1':
            raise ValueError('Unqualified candidate requires an explicit bounded diagnostic run')
        if approval.get('prediction_kind') in ('composite_pose', 'experimental_composite_pose'):
            self.close_grasp_model = SCQualifiedCloseCompositeObservation(os.environ['AIC_SC_CLOSE_GRASP_CHECKPOINT'], str(self.local.device), approval=os.environ['AIC_SC_CLOSE_GRASP_APPROVAL'], experimental_diagnostic=experimental)
        elif approval.get('prediction_kind') == 'orientation_only':
            self.close_grasp_model = SCQualifiedCloseOrientationObservation(os.environ['AIC_SC_CLOSE_GRASP_CHECKPOINT'], str(self.local.device), approval=os.environ['AIC_SC_CLOSE_GRASP_APPROVAL'], retained_translation_checkpoint=os.environ['AIC_LOCKED_GRASP_CHECKPOINT'])
        else:
            self.close_grasp_model = SCSpatialGraspObservation(os.environ['AIC_SC_CLOSE_GRASP_CHECKPOINT'], str(self.local.device), approval=os.environ['AIC_SC_CLOSE_GRASP_APPROVAL'])
        if not self.close_grasp_model.approved and (not experimental):
            raise ValueError('An independently qualified exact checkpoint is mandatory')
        self._trace(event='close_pose_scope', experimental_trial_only=experimental, model_globally_qualified=self.close_grasp_model.approved, geometry_input=False)
        self.servo_goal_translation = np.zeros(3)
        self.close_update_required = True
        self.close_networks_warmed = False

    @torch.inference_mode()
    def _warm_grasp(self):
        super()._warm_grasp()
        if self.close_networks_warmed:
            return
        device = self.local.device
        model = self.close_grasp_model
        for _ in range(3):
            if isinstance(model, SCQualifiedCloseCompositeObservation):
                model.position.model(torch.zeros(3, 3, 192, 192, device=device), camera_id=torch.eye(3, device=device))
                model.rotation.model(torch.zeros(1, 3, 3, 192, 192, device=device, dtype=torch.uint8))
            elif getattr(model, 'rotation_only', False):
                model.model(torch.zeros(1, 3, 3, 192, 192, device=device, dtype=torch.uint8))
            else:
                model.model(torch.zeros(3, 3, 192, 192, device=device), camera_id=torch.eye(3, device=device))
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        self.close_networks_warmed = True
        self._trace(event='close_grasp_networks_prewarmed', synthetic_pixels_used_for_pose=False, geometry_input=False)

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
            self.servo_goal_rotation = np.eye(3)
            self.servo_goal_translation = np.zeros(3)
            self._trace(event='close_pose_retry_alignment_retained', legacy_angular_search_disabled=True, geometry_input=False)
        return accepted

    def _visual_servo_insert(self, task, get_observation, move_robot, send_feedback, **kwargs):
        if self.local.task_vector(task)[0] != 1 and self.close_update_required:
            if not self._reestimate_close_grasp(get_observation, move_robot):
                return False
            self.close_update_required = False
        return self._close_pose_servo(task, get_observation, move_robot, send_feedback, **kwargs)

    def _close_pose_servo(self, task, get_observation, move_robot, send_feedback, **kwargs):
        vector = self.local.task_vector(task)
        port = self.local.visual_port.copy()
        inverse = np.linalg.inv(port)
        grasp = np.eye(4)
        grasp[:3, 3] = self.local.visual_body_tip
        grasp[:3, :3] = self.local.visual_body_connector_rotation
        inverse_grasp = np.linalg.inv(grasp)
        depth = 0.0465 if vector[0] == 1 else 0.0156
        depth += float(self.servo_goal_translation[2])
        command_depth = depth + (0.0004 if vector[0] != 1 else 0.0)
        seed, seed_stamp = self.local.governor_seed
        started = time.monotonic()
        first = None
        last = None
        stable = None
        best = -float('inf')
        last_progress = None
        reference = None
        integral = None
        governor = None
        reason = 'visual_servo_wall_budget'
        while time.monotonic() - started < 60 and time.monotonic() < self.episode_deadline - 2:
            observation = get_observation()
            if observation is None:
                time.sleep(0.002)
                continue
            stamp = self.local._control_sim_time(observation)
            if last is not None and stamp - last < 0.049:
                time.sleep(0.002)
                continue
            control_started = time.perf_counter()
            dt = 0.05 if last is None else min(0.1, stamp - last)
            last = stamp
            state = self.local.assembler.assemble_ros(observation)
            measured = matrix_from_xyz_xyzw(state[:7])
            point = inverse @ measured @ grasp
            angle = Rotation.from_matrix(self.servo_goal_rotation.T @ point[:3, :3]).magnitude()
            force = self._force(state)
            if first is None:
                first = stamp
                last_progress = stamp
                reference = np.array([self.servo_goal_translation[0], self.servo_goal_translation[1], point[2, 3]])
                integral = (inverse @ seed @ grasp)[:3, 3] - reference
                governor = InsertionReferenceGovernor(port, grasp, seed, measured, seed_stamp, ReferenceLimits(forward_lead_m=0.02, alignment_lateral_m=1.0, alignment_angle_rad=np.pi, contact_load_n=float('inf')))
                self._trace(event='visual_servo_started', sim_time=stamp, geometry_input=False, actor_owned=False, learned_actor_inference=False, required_seating_dwell_s=2.0, nominal_seating_depth_m=depth, command_depth_m=command_depth, source='frozen RGB port/grasp and public robot state only')
            if point[2, 3] > best + 0.0005:
                best = float(point[2, 3])
                last_progress = stamp
            if np.linalg.norm(state[26:29]) > 40 or (force >= 8 and self.mode == 'none'):
                reason = 'visual_servo_force_guard'
                break
            if stamp - first > 35:
                reason = 'visual_servo_simulation_budget'
                break
            if stamp - last_progress > 8:
                reason = 'visual_servo_no_measured_progress'
                break
            lateral = float(np.linalg.norm(point[:2, 3] - self.servo_goal_translation[:2]))
            aligned = bool(lateral <= 0.0005 and angle <= np.deg2rad(0.5))
            if aligned:
                reference[2] = min(command_depth, reference[2] + 0.006 * dt)
            integral = np.clip(integral + 0.6 * dt * (reference - point[:3, 3]), -0.015, 0.015)
            desired_point = np.eye(4)
            desired_point[:3, 3] = reference + integral
            desired_point[:3, :3] = self.servo_goal_rotation
            desired = port @ desired_point @ inverse_grasp
            issued, diagnostic = governor.step(desired, measured, stamp, force)
            if self.mode != 'none':
                self._supervise(state, issued)
                self._supervise_servo_stall(state, issued, point, stamp, force, depth)
            self._command(move_robot, issued, owner='image_only_insertion_servo_diagnostic')
            self._trace(event='visual_servo_step', sim_time=stamp, estimated_tip_port_m=point[:3, 3].tolist(), estimated_orientation_error_deg=float(np.degrees(angle)), reference_tip_port_m=reference.tolist(), integral_port_m=integral.tolist(), load_change_n=force, aligned=aligned, persistent_goal_translation_m=self.servo_goal_translation.tolist(), goal_orientation_offset_deg=np.rad2deg(Rotation.from_matrix(self.servo_goal_rotation).as_rotvec()).tolist(), complete_control_step_ms=(time.perf_counter() - control_started) * 1000, geometry_input=False, actor_owned=False, learned_actor_inference=False, governor=diagnostic)
            seated = point[2, 3] >= depth - 0.0002 and lateral <= 0.0005 and (angle <= np.deg2rad(0.75)) and (np.linalg.norm(state[7:10]) <= 0.002)
            stable = (stamp if stable is None else stable) if seated else None
            if stable is not None and stamp - stable >= 2.0:
                reason = 'visual_servo_public_estimated_seating'
                self._trace(event='visual_servo_finished', reason=reason, sim_time=stamp, geometry_input=False, physical_insertion_asserted=False)
                return True
        self._trace(event='visual_servo_finished', reason=reason, sim_time=last, geometry_input=False, physical_insertion_asserted=False)
        return False

class _RunVisualConnectorServoRecoveryV26(_RunVisualConnectorServoRecoveryV21):

    def _close_pose_servo(self, task, get_observation, move_robot, send_feedback, **kwargs):
        vector = self.local.task_vector(task)
        if vector[0] != 1:
            return super()._close_pose_servo(task, get_observation, move_robot, send_feedback, **kwargs)
        port = self.local.visual_port.copy()
        inverse = np.linalg.inv(port)
        grasp = np.eye(4)
        grasp[:3, 3] = self.local.visual_body_tip
        grasp[:3, :3] = self.local.visual_body_connector_rotation
        inverse_grasp = np.linalg.inv(grasp)
        depth = 0.0465 if vector[0] == 1 else 0.0156
        depth += float(self.servo_goal_translation[2])
        command_depth = depth + (0.0004 if vector[0] != 1 else 0.0)
        seed, seed_stamp = self.local.governor_seed
        started = time.monotonic()
        first = None
        last = None
        stable = None
        best = -float('inf')
        last_progress = None
        reference = None
        integral = None
        governor = None
        reason = 'visual_servo_wall_budget'
        soft_force_since = None
        last_issued = np.asarray(seed).copy()
        while time.monotonic() - started < 60 and time.monotonic() < self.episode_deadline - 2:
            observation = get_observation()
            if observation is None:
                time.sleep(0.002)
                continue
            stamp = self.local._control_sim_time(observation)
            if last is not None and stamp - last < 0.049:
                time.sleep(0.002)
                continue
            control_started = time.perf_counter()
            dt = 0.05 if last is None else min(0.1, stamp - last)
            last = stamp
            state = self.local.assembler.assemble_ros(observation)
            measured = matrix_from_xyz_xyzw(state[:7])
            point = inverse @ measured @ grasp
            angle = Rotation.from_matrix(self.servo_goal_rotation.T @ point[:3, :3]).magnitude()
            force = self._force(state)
            if first is None:
                first = stamp
                last_progress = stamp
                reference = np.array([self.servo_goal_translation[0], self.servo_goal_translation[1], point[2, 3]])
                integral = (inverse @ seed @ grasp)[:3, 3] - reference
                governor = InsertionReferenceGovernor(port, grasp, seed, measured, seed_stamp, ReferenceLimits(forward_lead_m=0.02, alignment_lateral_m=1.0, alignment_angle_rad=np.pi, contact_load_n=float('inf')))
                self._trace(event='visual_servo_started', sim_time=stamp, geometry_input=False, actor_owned=False, learned_actor_inference=False, required_seating_dwell_s=getattr(self, 'seating_dwell_s', 2.0), nominal_seating_depth_m=depth, command_depth_m=command_depth, source='frozen RGB port/grasp and public robot state only')
            if point[2, 3] > best + 0.0005:
                best = float(point[2, 3])
                last_progress = stamp
            if np.linalg.norm(state[26:29]) > 40:
                reason = 'visual_servo_raw_force_guard'
                break
            if force >= 8 and self.mode == 'none':
                near_seat = point[2, 3] >= depth - 0.0005 and np.linalg.norm(point[:2, 3]) <= 0.0005 and (angle <= np.deg2rad(0.75)) and (np.linalg.norm(state[7:10]) <= 0.005)
                if not near_seat:
                    reason = 'visual_servo_force_guard_outside_seat'
                    break
                if soft_force_since is None:
                    soft_force_since = stamp
                self._command(move_robot, last_issued, owner='SFP_seating_force_pause')
                self._trace(event='SFP_seating_force_pause', sim_time=stamp, load_change_n=force, raw_force_n=float(np.linalg.norm(state[26:29])), elapsed_s=stamp - soft_force_since, forward_motion_requested=False, held_api_reference=last_issued.tolist(), geometry_input=False)
                if stamp - soft_force_since >= 0.099:
                    reason = 'visual_servo_persistent_seat_load_guard'
                    break
                continue
            if soft_force_since is not None:
                self._trace(event='SFP_seating_force_spike_cleared', sim_time=stamp, load_change_n=force, paused_s=stamp - soft_force_since, geometry_input=False)
                soft_force_since = None
            if stamp - first > 35:
                reason = 'visual_servo_simulation_budget'
                break
            if stamp - last_progress > 8:
                reason = 'visual_servo_no_measured_progress'
                break
            lateral = float(np.linalg.norm(point[:2, 3] - self.servo_goal_translation[:2]))
            aligned = bool(lateral <= 0.0005 and angle <= np.deg2rad(0.5))
            if aligned:
                reference[2] = min(command_depth, reference[2] + getattr(self, 'insertion_axial_speed_m_s', 0.006) * dt)
            integral = np.clip(integral + 0.6 * dt * (reference - point[:3, 3]), -0.015, 0.015)
            desired_point = np.eye(4)
            desired_point[:3, 3] = reference + integral
            desired_point[:3, :3] = self.servo_goal_rotation
            desired = port @ desired_point @ inverse_grasp
            issued, diagnostic = governor.step(desired, measured, stamp, force)
            if self.mode != 'none':
                self._supervise(state, issued)
                self._supervise_servo_stall(state, issued, point, stamp, force, depth)
            self._command(move_robot, issued, owner='image_only_insertion_servo_diagnostic')
            last_issued = issued.copy()
            self._trace(event='visual_servo_step', sim_time=stamp, estimated_tip_port_m=point[:3, 3].tolist(), estimated_orientation_error_deg=float(np.degrees(angle)), reference_tip_port_m=reference.tolist(), integral_port_m=integral.tolist(), load_change_n=force, aligned=aligned, persistent_goal_translation_m=self.servo_goal_translation.tolist(), goal_orientation_offset_deg=np.rad2deg(Rotation.from_matrix(self.servo_goal_rotation).as_rotvec()).tolist(), complete_control_step_ms=(time.perf_counter() - control_started) * 1000, geometry_input=False, actor_owned=False, learned_actor_inference=False, governor=diagnostic)
            seated = point[2, 3] >= depth - 0.0002 and lateral <= 0.0005 and (angle <= np.deg2rad(0.75)) and (np.linalg.norm(state[7:10]) <= 0.002)
            stable = (stamp if stable is None else stable) if seated else None
            if stable is not None and stamp - stable >= getattr(self, 'seating_dwell_s', 2.0):
                reason = 'visual_servo_public_estimated_seating'
                self._trace(event='visual_servo_finished', reason=reason, sim_time=stamp, geometry_input=False, physical_insertion_asserted=False)
                return True
        self._trace(event='visual_servo_finished', reason=reason, sim_time=last, geometry_input=False, physical_insertion_asserted=False)
        return False

class _RunVisualConnectorServoContextV28(_RunVisualConnectorServoRecoveryV26):

    def _lock(self, task_vector, get_observation, move_robot, *, fresh_context):
        self.far_grasp_frames = []
        self.far_orientation = None
        return super()._lock(task_vector, get_observation, move_robot, fresh_context=fresh_context)

    def _fresh_report(self, observation, task_vector):
        report, images, state, stamp = super()._fresh_report(observation, task_vector)
        if task_vector[1] == 1 and report['valid'] and (np.linalg.norm(state[7:10]) <= 0.005) and (np.linalg.norm(state[10:13]) <= 0.03):
            stamps = [getattr(observation, c + '_image').header.stamp for c in ('center', 'left', 'right')]
            stamps = tuple((x.sec * 1000000000 + x.nanosec for x in stamps))
            if not self.far_grasp_frames or all((a > b for a, b in zip(stamps, self.far_grasp_frames[-1][0]))):
                self.far_grasp_frames = (self.far_grasp_frames + [(stamps, {c: x.copy() for c, x in images.items()})])[-3:]
        return (report, images, state, stamp)

    def _prepare_far_orientation(self):
        started = time.perf_counter()
        self.far_orientation = None
        if len(self.far_grasp_frames) != 3:
            self._trace(event='far_orientation_context_unavailable', fallback='near_only', geometry_input=False)
            return (time.perf_counter() - started) * 1000
        model = getattr(self.close_grasp_model, 'rotation', None)
        if model is None:
            raise ValueError('This context diagnostic requires the explicit composite RGB model')
        rotations = []
        for stamps, images in self.far_grasp_frames:
            result = model.predict(images, visual_tip_position_tcp_m=self.local.visual_body_tip)
            rotations.append(np.asarray(result['orientation_for_diagnostic_only']))
        span = max((float(np.degrees(Rotation.from_matrix(a.T @ b).magnitude())) for a in rotations for b in rotations))
        if span <= 0.5:
            self.far_orientation = Rotation.from_matrix(rotations).mean().as_matrix()
        self._trace(event='far_grasp_history_encoded_at25mm', image_stamps_ns=[x[0] for x in self.far_grasp_frames], history_not_current_observation=True, rotation_span_deg=span, accepted=span <= 0.5, same_stronger_RGB_model=True, old_RGB_orientation_not_fused=True, physical_grasp_rigidity_assumed=True, geometry_input=False, elapsed_ms=(time.perf_counter() - started) * 1000)
        return (time.perf_counter() - started) * 1000

    def _reestimate_close_grasp(self, get_observation, move_robot):
        far_ms = self._prepare_far_orientation()
        reference, reference_stamp = self.local.governor_seed
        reference = reference.copy()
        old_position = np.asarray(self.local.visual_body_tip).copy()
        old_rotation = np.asarray(self.local.visual_body_connector_rotation).copy()
        consensus = CloseGraspConsensus()
        first_candidate = None
        started = time.monotonic()
        last_sim = None
        while time.monotonic() - started < 10 and time.monotonic() < self.episode_deadline - 2:
            observation = get_observation()
            if observation is None:
                time.sleep(0.002)
                continue
            stamp = self.local._control_sim_time(observation)
            if last_sim is not None and stamp - last_sim < 0.099:
                time.sleep(0.002)
                continue
            last_sim = stamp
            control_started = time.perf_counter() - far_ms / 1000
            state = self.local.assembler.assemble_ros(observation)
            self._command(move_robot, reference, owner='close_grasp_observation_hold')
            if self._force(state) >= 8 or np.linalg.norm(state[26:29]) > 40:
                self._trace(event='close_grasp_update_rejected', reason='loaded_observation_hold', geometry_input=False)
                return False
            point = np.linalg.inv(self.local.visual_port) @ matrix_from_xyz_xyzw(state[:7])
            tip = point[:3, 3] + point[:3, :3] @ old_position
            if not -0.03 <= tip[2] <= -0.02:
                self._trace(event='close_grasp_update_rejected', reason='outside_validated_near_view', estimated_axial_m=float(tip[2]), geometry_input=False)
                return False
            if np.linalg.norm(state[7:10]) > 0.002 or np.linalg.norm(state[10:13]) > 0.03:
                consensus = CloseGraspConsensus()
                first_candidate = None
                continue
            stamps = [getattr(observation, camera + '_image').header.stamp for camera in ('center', 'left', 'right')]
            stamps = [s.sec * 1000000000 + s.nanosec for s in stamps]
            ages = [stamp - s * 1e-09 for s in stamps]
            if min(ages) < -0.001 or max(ages) > 0.1 or max(stamps) - min(stamps) > 50000000:
                self._trace(event='close_grasp_frame_rejected', reason='stale_or_unsynchronized_RGB', ages_s=ages, geometry_input=False)
                continue
            images = {c: _RunRuleTransport.native(getattr(observation, c + '_image')) for c in ('center', 'left', 'right')}
            if getattr(self.close_grasp_model, 'rotation_only', False):
                prediction = self.close_grasp_model.predict(images, visual_tip_position_tcp_m=old_position)
            else:
                prediction = self.close_grasp_model.predict(images)
            candidate = consensus.add(stamps, prediction['predicted_tip_position_tcp_m'], prediction['orientation_for_diagnostic_only'])
            self._trace(event='close_grasp_observation', sim_time=stamp, image_stamps_ns=stamps, consensus=candidate, prediction=prediction, complete_close_estimate_step_ms=(time.perf_counter() - control_started) * 1000, geometry_input=False)
            far_ms = 0.0
            if not candidate['accepted']:
                continue
            if self.far_orientation is not None:
                near = np.asarray(candidate['rotation'])
                difference = float(np.degrees(Rotation.from_matrix(self.far_orientation.T @ near).magnitude()))
                if difference <= 1.0:
                    candidate['rotation'] = Rotation.from_matrix([self.far_orientation, near]).mean().as_matrix().tolist()
                    candidate['orientation_context_fusion'] = 'equal_stronger_model_far_and_near'
                    candidate['far_near_difference_deg'] = difference
                    candidate['translation_from_near_only'] = True
                else:
                    self._trace(event='far_near_orientation_fusion_rejected', difference_deg=difference, fallback='near_only', geometry_input=False)
            change = compare_grasp_update(old_position, old_rotation, candidate)
            if not change['accept_update']:
                if first_candidate is None:
                    first_candidate = candidate
                    consensus = CloseGraspConsensus()
                    consensus.last_stamps = tuple(stamps)
                    self._trace(event='close_grasp_large_disagreement_second_window', **change, geometry_input=False)
                    continue
                repeat_mm = float(np.linalg.norm(np.asarray(first_candidate['position_m']) - np.asarray(candidate['position_m'])) * 1000)
                repeat_deg = float(np.degrees(Rotation.from_matrix(np.asarray(first_candidate['rotation']).T @ np.asarray(candidate['rotation'])).magnitude()))
                confirmed = bool(change['translation_change_mm'] <= 1.0 and change['rotation_change_deg'] <= 4.0 and (repeat_mm <= 0.2) and (repeat_deg <= 0.5))
                self._trace(event='close_grasp_two_window_confirmation', accepted=confirmed, repeat_translation_mm=repeat_mm, repeat_rotation_deg=repeat_deg, old_to_near_rotation_deg=change['rotation_change_deg'], geometry_input=False)
                if not confirmed:
                    self._trace(event='close_grasp_update_rejected', reason='persistent_large_disagreement_reacquisition_required', first_candidate=first_candidate, candidate=candidate, **change, geometry_input=False)
                    return False
                change = dict(change, accept_update=True, two_window_confirmation=True)
            self.local.visual_body_tip = np.asarray(candidate['position_m'])
            self.local.visual_body_connector_rotation = np.asarray(candidate['rotation'])
            self.locked_connector_rotation = self.local.visual_body_connector_rotation.copy()
            self.local.clear_pending_actions()
            self.local.governor = None
            self.local.governor_seed = (reference, stamp)
            self.servo_goal_rotation = np.eye(3)
            self.servo_goal_translation = np.zeros(3)
            self.servo_stall_history = []
            self._trace(event='close_grasp_update_accepted', sim_time=stamp, **change, old_position_tcp_m=old_position.tolist(), old_rotation_tcp=old_rotation.tolist(), candidate=candidate, unchanged_api_reference=reference.tolist(), port_pose_unchanged=True, locked_within_attempt=True, old_estimator_calibration_reused=False, geometry_input=False)
            return True
        self._trace(event='close_grasp_update_rejected', reason='consistent_fresh_observations_unavailable', geometry_input=False)
        return False

class _RunVisualConnectorServoAlignedV29(_RunVisualConnectorServoContextV28):

    def _close_pose_servo(self, task, get_observation, move_robot, send_feedback, **kwargs):
        vector = self.local.task_vector(task)
        if vector[0] == 1:
            return super()._close_pose_servo(task, get_observation, move_robot, send_feedback, **kwargs)
        port = self.local.visual_port.copy()
        inverse = np.linalg.inv(port)
        grasp = np.eye(4)
        grasp[:3, 3] = self.local.visual_body_tip
        grasp[:3, :3] = self.local.visual_body_connector_rotation
        inverse_grasp = np.linalg.inv(grasp)
        depth = 0.0465 if vector[0] == 1 else 0.0156
        depth += float(self.servo_goal_translation[2])
        command_depth = depth + (0.0004 if vector[0] != 1 else 0.0)
        seed, seed_stamp = self.local.governor_seed
        started = time.monotonic()
        first = None
        last = None
        stable = None
        best = -float('inf')
        last_progress = None
        reference = None
        integral = None
        governor = None
        reason = 'visual_servo_wall_budget'
        alignment_streak = 0
        alignment_latched = False
        while time.monotonic() - started < 60 and time.monotonic() < self.episode_deadline - 2:
            observation = get_observation()
            if observation is None:
                time.sleep(0.002)
                continue
            stamp = self.local._control_sim_time(observation)
            if last is not None and stamp - last < 0.049:
                time.sleep(0.002)
                continue
            control_started = time.perf_counter()
            dt = 0.05 if last is None else min(0.1, stamp - last)
            last = stamp
            state = self.local.assembler.assemble_ros(observation)
            measured = matrix_from_xyz_xyzw(state[:7])
            point = inverse @ measured @ grasp
            angle = Rotation.from_matrix(self.servo_goal_rotation.T @ point[:3, :3]).magnitude()
            force = self._force(state)
            if first is None:
                first = stamp
                last_progress = stamp
                reference = np.array([self.servo_goal_translation[0], self.servo_goal_translation[1], point[2, 3]])
                integral = (inverse @ seed @ grasp)[:3, 3] - reference
                governor = InsertionReferenceGovernor(port, grasp, seed, measured, seed_stamp, ReferenceLimits(forward_lead_m=0.02, alignment_lateral_m=1.0, alignment_angle_rad=np.pi, contact_load_n=float('inf')))
                self._trace(event='visual_servo_started', sim_time=stamp, geometry_input=False, actor_owned=False, learned_actor_inference=False, required_seating_dwell_s=2.0, nominal_seating_depth_m=depth, command_depth_m=command_depth, source='frozen RGB port/grasp and public robot state only')
            if point[2, 3] > best + 0.0005:
                best = float(point[2, 3])
                last_progress = stamp
            if np.linalg.norm(state[26:29]) > 40 or (force >= 8 and self.mode == 'none'):
                reason = 'visual_servo_force_guard'
                break
            if stamp - first > 35:
                reason = 'visual_servo_simulation_budget'
                break
            if stamp - last_progress > 8:
                reason = 'visual_servo_no_measured_progress'
                break
            lateral = float(np.linalg.norm(point[:2, 3] - self.servo_goal_translation[:2]))
            alignment_streak = alignment_streak + 1 if lateral <= 0.0001 and angle <= np.deg2rad(0.5) else 0
            if lateral > 0.00015 or angle > np.deg2rad(0.5):
                alignment_latched = False
            elif alignment_streak >= 3:
                alignment_latched = True
            aligned = alignment_latched
            if aligned:
                reference[2] = min(command_depth, reference[2] + 0.006 * dt)
            integral = np.clip(integral + 0.6 * dt * (reference - point[:3, 3]), -0.015, 0.015)
            desired_point = np.eye(4)
            desired_point[:3, 3] = reference + integral
            desired_point[:3, :3] = self.servo_goal_rotation
            desired = port @ desired_point @ inverse_grasp
            issued, diagnostic = governor.step(desired, measured, stamp, force)
            if self.mode != 'none':
                self._supervise(state, issued)
                self._supervise_servo_stall(state, issued, point, stamp, force, depth)
            self._command(move_robot, issued, owner='image_only_insertion_servo_diagnostic')
            self._trace(event='visual_servo_step', sim_time=stamp, estimated_tip_port_m=point[:3, 3].tolist(), estimated_orientation_error_deg=float(np.degrees(angle)), reference_tip_port_m=reference.tolist(), integral_port_m=integral.tolist(), load_change_n=force, aligned=aligned, advance_gate_entry_lateral_mm=0.1, advance_gate_exit_lateral_mm=0.15, alignment_streak=alignment_streak, persistent_goal_translation_m=self.servo_goal_translation.tolist(), goal_orientation_offset_deg=np.rad2deg(Rotation.from_matrix(self.servo_goal_rotation).as_rotvec()).tolist(), complete_control_step_ms=(time.perf_counter() - control_started) * 1000, geometry_input=False, actor_owned=False, learned_actor_inference=False, governor=diagnostic)
            seated = point[2, 3] >= depth - 0.0002 and lateral <= 0.0005 and (angle <= np.deg2rad(0.75)) and (np.linalg.norm(state[7:10]) <= 0.002)
            stable = (stamp if stable is None else stable) if seated else None
            if stable is not None and stamp - stable >= 2.0:
                reason = 'visual_servo_public_estimated_seating'
                self._trace(event='visual_servo_finished', reason=reason, sim_time=stamp, geometry_input=False, physical_insertion_asserted=False)
                return True
        self._trace(event='visual_servo_finished', reason=reason, sim_time=last, geometry_input=False, physical_insertion_asserted=False)
        return False

class _RunVisualConnectorServoBiasSeedV31(_RunVisualConnectorServoAlignedV29):

    def _close_pose_servo(self, task, get_observation, move_robot, send_feedback, **kwargs):
        vector = self.local.task_vector(task)
        if vector[0] == 1:
            return super()._close_pose_servo(task, get_observation, move_robot, send_feedback, **kwargs)
        port = self.local.visual_port.copy()
        inverse = np.linalg.inv(port)
        grasp = np.eye(4)
        grasp[:3, 3] = self.local.visual_body_tip
        grasp[:3, :3] = self.local.visual_body_connector_rotation
        inverse_grasp = np.linalg.inv(grasp)
        depth = 0.0465 if vector[0] == 1 else 0.0156
        depth += float(self.servo_goal_translation[2])
        command_depth = depth + (0.0004 if vector[0] != 1 else 0.0)
        seed, seed_stamp = self.local.governor_seed
        started = time.monotonic()
        first = None
        last = None
        stable = None
        best = -float('inf')
        last_progress = None
        reference = None
        integral = None
        governor = None
        reason = 'visual_servo_wall_budget'
        alignment_streak = 0
        alignment_latched = False
        while time.monotonic() - started < 60 and time.monotonic() < self.episode_deadline - 2:
            observation = get_observation()
            if observation is None:
                time.sleep(0.002)
                continue
            stamp = self.local._control_sim_time(observation)
            if last is not None and stamp - last < 0.049:
                time.sleep(0.002)
                continue
            control_started = time.perf_counter()
            dt = 0.05 if last is None else min(0.1, stamp - last)
            last = stamp
            state = self.local.assembler.assemble_ros(observation)
            measured = matrix_from_xyz_xyzw(state[:7])
            point = inverse @ measured @ grasp
            angle = Rotation.from_matrix(self.servo_goal_rotation.T @ point[:3, :3]).magnitude()
            force = self._force(state)
            if first is None:
                first = stamp
                last_progress = stamp
                reference = np.array([self.servo_goal_translation[0], self.servo_goal_translation[1], point[2, 3]])
                integral = (inverse @ seed @ grasp)[:3, 3] - reference
                old_integral = integral.copy()
                bias_port = inverse[:3, :3] @ np.asarray(state[13:16])
                integral[:2] = np.clip(bias_port[:2], -0.015, 0.015)
                self._trace(event='lateral_integral_seed_from_public_tracking_error', sim_time=stamp, previous_integral_port_m=old_integral.tolist(), new_integral_port_m=integral.tolist(), public_TCP_error_base_m=state[13:16].tolist(), governor_still_seeded_from_same_API_reference=True, axial_seed_unchanged=True, force_baseline_retared=False, geometry_input=False)
                governor = InsertionReferenceGovernor(port, grasp, seed, measured, seed_stamp, ReferenceLimits(forward_lead_m=0.02, alignment_lateral_m=1.0, alignment_angle_rad=np.pi, contact_load_n=float('inf')))
                self._trace(event='visual_servo_started', sim_time=stamp, geometry_input=False, actor_owned=False, learned_actor_inference=False, required_seating_dwell_s=getattr(self, 'seating_dwell_s', 2.0), nominal_seating_depth_m=depth, command_depth_m=command_depth, source='frozen RGB port/grasp and public robot state only')
            if point[2, 3] > best + 0.0005:
                best = float(point[2, 3])
                last_progress = stamp
            if np.linalg.norm(state[26:29]) > 40 or (force >= 8 and self.mode == 'none'):
                reason = 'visual_servo_force_guard'
                break
            if stamp - first > 35:
                reason = 'visual_servo_simulation_budget'
                break
            if stamp - last_progress > 8:
                reason = 'visual_servo_no_measured_progress'
                break
            lateral = float(np.linalg.norm(point[:2, 3] - self.servo_goal_translation[:2]))
            alignment_streak = alignment_streak + 1 if lateral <= 0.0001 and angle <= np.deg2rad(0.5) else 0
            if lateral > 0.00015 or angle > np.deg2rad(0.5):
                alignment_latched = False
            elif alignment_streak >= 3:
                alignment_latched = True
            aligned = alignment_latched
            if aligned:
                reference[2] = min(command_depth, reference[2] + getattr(self, 'insertion_axial_speed_m_s', 0.006) * dt)
            integral = np.clip(integral + 0.6 * dt * (reference - point[:3, 3]), -0.015, 0.015)
            desired_point = np.eye(4)
            desired_point[:3, 3] = reference + integral
            desired_point[:3, :3] = self.servo_goal_rotation
            desired = port @ desired_point @ inverse_grasp
            issued, diagnostic = governor.step(desired, measured, stamp, force)
            if self.mode != 'none':
                self._supervise(state, issued)
                self._supervise_servo_stall(state, issued, point, stamp, force, depth)
            self._command(move_robot, issued, owner='image_only_insertion_servo_diagnostic')
            self._trace(event='visual_servo_step', sim_time=stamp, estimated_tip_port_m=point[:3, 3].tolist(), estimated_orientation_error_deg=float(np.degrees(angle)), reference_tip_port_m=reference.tolist(), integral_port_m=integral.tolist(), load_change_n=force, aligned=aligned, advance_gate_entry_lateral_mm=0.1, advance_gate_exit_lateral_mm=0.15, alignment_streak=alignment_streak, persistent_goal_translation_m=self.servo_goal_translation.tolist(), goal_orientation_offset_deg=np.rad2deg(Rotation.from_matrix(self.servo_goal_rotation).as_rotvec()).tolist(), complete_control_step_ms=(time.perf_counter() - control_started) * 1000, geometry_input=False, actor_owned=False, learned_actor_inference=False, governor=diagnostic)
            seated = point[2, 3] >= depth - 0.0002 and lateral <= 0.0005 and (angle <= np.deg2rad(0.75)) and (np.linalg.norm(state[7:10]) <= 0.002)
            stable = (stamp if stable is None else stable) if seated else None
            if stable is not None and stamp - stable >= getattr(self, 'seating_dwell_s', 2.0):
                reason = 'visual_servo_public_estimated_seating'
                self._trace(event='visual_servo_finished', reason=reason, sim_time=stamp, geometry_input=False, physical_insertion_asserted=False)
                return True
        self._trace(event='visual_servo_finished', reason=reason, sim_time=last, geometry_input=False, physical_insertion_asserted=False)
        return False

class _RunVisualConnectorServoStiffV32(_RunVisualConnectorServoBiasSeedV31):
    lateral_stiffness = 240.0

    def _execute(self, task, get_observation, move_robot, send_feedback, **kwargs):
        sc = self.local.task_vector(task)[1] == 1
        stiffness = self.lateral_stiffness if sc else 90.0
        self.local.cartesian_stiffness = [stiffness, stiffness, 90.0, 50.0, 50.0, 50.0]
        self._trace(event='SC_lateral_stiffness_diagnostic', family='sc' if sc else 'sfp', stiffness_base=self.local.cartesian_stiffness, damping=self.local.cartesian_damping, axial_stiffness_unchanged=True, force_guards_unchanged=True, geometry_input=False)
        return super()._execute(task, get_observation, move_robot, send_feedback, **kwargs)

class _RunVisualConnectorServoStiffV33(_RunVisualConnectorServoStiffV32):
    lateral_stiffness = 500.0

class _RunVisualConnectorServoRecoveryFastV34(_RunVisualConnectorServoStiffV33):

    def _warm_grasp(self):
        super()._warm_grasp()
        self.grasp_jpeg_cache = SharedGraspTrainingJPEG()
        self.grasp = SharedJPEGEstimator(self.grasp, self.grasp_jpeg_cache, begin=False)
        for family, model in list(self.connector_models.items()):
            if isinstance(model, SharedJPEGEstimator):
                model = model.estimator
            self.connector_models[family] = SharedJPEGEstimator(model, self.grasp_jpeg_cache, begin=True)
        self._trace(event='shared_native_grasp_jpeg_enabled', quality=86, codec_passes_per_triplet=1, model_weights_changed=False, geometry_input=False)

class _RunVisualConnectorServoSCEmptyDiagnosticV38(_RunVisualConnectorServoRecoveryFastV34):
    insertion_axial_speed_m_s = 0.012

    def _make_rule_transport(self):
        return self.cached_transport

class _RunVisualConnectorServoOldPoseEfficientV39(_RunVisualConnectorServoSCEmptyDiagnosticV38):

    def _execute(self, task, get_observation, move_robot, send_feedback, **kwargs):
        return _RunVisualConnectorServoRecoveryFastV34._execute(self, task, get_observation, move_robot, send_feedback, **kwargs)

class _RunVisualConnectorServoWindowV40(_RunVisualConnectorServoOldPoseEfficientV39):

    def _command(self, move_robot, matrix, *, owner):
        if owner == 'initial_free_hold':
            self.initial_hold_reference = np.asarray(matrix).copy()
        return super()._command(move_robot, matrix, owner=owner)

    def _initial_force_reference(self, get_observation, move_robot):
        if not super()._initial_force_reference(get_observation, move_robot):
            return False
        if self.scene_task_code[1] != 1:
            return True
        if self.initial_hold_reference is None:
            return False
        consensus = EmptySceneConsensus(packet_camera_timing=getattr(self, 'use_packet_camera_timing', False))
        started = time.monotonic()
        last = -float('inf')
        first = None
        while time.monotonic() - started < 3:
            observation = get_observation()
            if observation is None:
                time.sleep(0.002)
                continue
            stamp = self.local._control_sim_time(observation)
            if stamp - last < 0.099:
                time.sleep(0.002)
                continue
            last = stamp
            first = stamp if first is None else first
            if stamp - first > 2:
                return True
            tick = time.perf_counter()
            state = self.local.assembler.assemble_ros(observation)
            self._command(move_robot, self.initial_hold_reference, owner='empty_row_initial_observation_hold')
            if self._force(state) >= 8 or np.linalg.norm(state[26:29]) > 40:
                self._trace(event='empty_row_scene_window_rejected', sim_time=stamp, reason='loaded_hold', geometry_input=False)
                return False
            images = {c: _RunRuleTransport.native(getattr(observation, c + '_image')) for c in ('center', 'left', 'right')}
            stamps = [getattr(observation, c + '_image').header.stamp.sec * 1000000000 + getattr(observation, c + '_image').header.stamp.nanosec for c in ('center', 'left', 'right')]
            prediction = self.cached_transport.empty_row.predict(images)
            receipt = consensus.add(stamp_s=stamp, image_stamps_ns=stamps, tcp_pose=state[:7], linear_speed=np.linalg.norm(state[7:10]), angular_speed=np.linalg.norm(state[10:13]), clear=prediction['clear'], task_code=self.scene_task_code)
            self._trace(event='empty_row_stationary_scene_check', sim_time=stamp, **prediction, consistent_observations=consensus.count, image_ages_s=(stamp - np.array(stamps) * 1e-09).tolist(), complete_scene_check_ms=(time.perf_counter() - tick) * 1000, geometry_input=False)
            if receipt:
                self.cached_transport.pending_initial_receipt = receipt
                return True
            if prediction['score'] < 0.2:
                return True
        return True

    def _execute(self, task, get_observation, move_robot, send_feedback, **kwargs):
        self.scene_task_code = self.local.task_vector(task)
        self.cached_transport.pending_initial_receipt = None
        try:
            return super()._execute(task, get_observation, move_robot, send_feedback, **kwargs)
        finally:
            self.cached_transport.pending_initial_receipt = None
            self.scene_task_code = None

class _RunVisualConnectorServoStableApproachV41(_RunVisualConnectorServoWindowV40):
    approach_antiwindup = True

class _RunVisualConnectorServoSFPStableV42(_RunVisualConnectorServoStableApproachV41):

    def _warm_grasp(self):
        super()._warm_grasp()
        self.far_context_prepared = False

    def _prepare_far_orientation(self):
        if self.far_context_prepared:
            return 0.0
        elapsed = super()._prepare_far_orientation()
        self.far_context_prepared = True
        return elapsed

    def _approach_tcp_rotation(self, port):
        rotation = super()._approach_tcp_rotation(port)
        if self.scene_task_code[1] == 1:
            elapsed = self._prepare_far_orientation()
            self._trace(event='far_context_prepared_before_approach', complete_preparation_ms=elapsed, source='same_three_retained_far_RGB_observations', geometry_input=False)
        return rotation

    def _close_pose_servo(self, task, get_observation, move_robot, send_feedback, **kwargs):
        if self.local.task_vector(task)[0] != 1:
            return super()._close_pose_servo(task, get_observation, move_robot, send_feedback, **kwargs)
        original = list(self.local.cartesian_stiffness)
        self.local.cartesian_stiffness = [500.0, 500.0, 90.0, 50.0, 50.0, 50.0]
        self._trace(event='SFP_lateral_stiffness_diagnostic', stiffness_base=self.local.cartesian_stiffness, damping=self.local.cartesian_damping, axial_stiffness_unchanged=True, force_guards_unchanged=True, geometry_input=False)
        try:
            return super()._close_pose_servo(task, get_observation, move_robot, send_feedback, **kwargs)
        finally:
            self.local.cartesian_stiffness = original

class _RunVisualConnectorServoFastApproachV43(_RunVisualConnectorServoSFPStableV42):

    def __init__(self, parent_node):
        _RunVisualConnectorServoRecoveryFastV34.__init__(self, parent_node)
        self.cached_transport = self._make_cached_transport(parent_node)
        bundle = self.cached_transport.spec['bundle']
        for family in ('sfp', 'sc'):
            self.cached_transport.components[family] = RequestedNativePortPose(bundle['backbone'], bundle[family + '_coarse'], bundle[family + '_native_refiner'], bundle['mount_priors'], family=family, device=str(self.local.device), maximum_shift_px=64.0, maximum_reprojection_px=2.5 if family == 'sc' else 2.0, minimum_projection_probability=0.9 if family == 'sc' else 0.95)
        self.scene_task_code = None
        self.initial_hold_reference = None

    def _approach_speed(self, axial):
        blend = float(np.clip((-axial - 0.03) / 0.05, 0.0, 1.0))
        blend = blend * blend * (3.0 - 2.0 * blend)
        return 0.04 + 0.04 * blend

    def _lock(self, task_vector, get_observation, move_robot, *, fresh_context):
        original = list(self.local.cartesian_stiffness)
        if task_vector[0] == 1:
            self.local.cartesian_stiffness = [500.0, 500.0, 90.0, 50.0, 50.0, 50.0]
        try:
            return super()._lock(task_vector, get_observation, move_robot, fresh_context=fresh_context)
        finally:
            self.local.cartesian_stiffness = original

    def _close_pose_servo(self, task, get_observation, move_robot, send_feedback, **kwargs):
        self.insertion_axial_speed_m_s = 0.024 if self.local.task_vector(task)[0] == 1 else 0.012
        return super()._close_pose_servo(task, get_observation, move_robot, send_feedback, **kwargs)

class _RunVisualConnectorServoDirectSFPV44(_RunVisualConnectorServoFastApproachV43):
    SFP_minimum_shared_tip_standoff_m = 0.07

class _RunVisualConnectorServoAcquireInPlaceV45(_RunVisualConnectorServoDirectSFPV44):
    SFP_maximum_shared_tip_lateral_m = 0.06

    def _lock(self, task_vector, get_observation, move_robot, *, fresh_context):
        observation = get_observation()
        if observation is None:
            return False
        reference = observation.controller_state.reference_tcp_pose
        position, q = (reference.position, reference.orientation)
        self.fresh_public_reference = matrix_from_xyz_xyzw([position.x, position.y, position.z, q.x, q.y, q.z, q.w])
        measured = matrix_from_xyz_xyzw(self.local.assembler.assemble_ros(observation)[:7])
        error = np.linalg.inv(measured) @ self.fresh_public_reference
        if np.linalg.norm(error[:3, 3]) > 0.04 or Rotation.from_matrix(error[:3, :3]).magnitude() > np.deg2rad(5.0):
            self.fresh_public_reference = None
            self._trace(event='fresh_public_reference_rejected', geometry_input=False)
            return False
        self._trace(event='fresh_acquisition_public_API_reference_preserved', source='public_controller_reference_tcp_pose', reference=self.fresh_public_reference.tolist(), geometry_input=False)
        try:
            return super()._lock(task_vector, get_observation, move_robot, fresh_context=fresh_context)
        finally:
            self.fresh_public_reference = None

    def _command(self, move_robot, matrix, *, owner):
        if owner == 'fresh_visual_acquisition_hold' and getattr(self, 'fresh_public_reference', None) is not None:
            matrix = self.fresh_public_reference
        return super()._command(move_robot, matrix, owner=owner)

class _RunVisualConnectorServoAcquireInPlaceV46(_RunVisualConnectorServoAcquireInPlaceV45):
    pass

class _RunVisualConnectorServoPersistentApproachV47(_RunVisualConnectorServoAcquireInPlaceV46):
    persistent_approach_orientation = True

class _RunVisualConnectorServoPersistentApproachV48(_RunVisualConnectorServoPersistentApproachV47):

    def _approach_tcp_rotation(self, port):
        reference = self.fresh_public_reference
        if reference is None:
            raise ValueError('Fresh public API hold must precede persistent approach')
        self.local.governor_seed = (reference.copy(), self.local.last_observation_stamp)
        self._trace(event='persistent_approach_seed_from_public_hold', source='public_controller_reference_tcp_pose', previous_episode_seed_reused=False, geometry_input=False)
        return super()._approach_tcp_rotation(port)

class _RunVisualConnectorServoDirectEmptySCV49(_RunVisualConnectorServoPersistentApproachV48):

    def _make_cached_transport(self, parent_node):
        return _RunObservedDirectEmptySCV49(parent_node)

class _RunVisualConnectorServoFreeApproachV50(_RunVisualConnectorServoDirectEmptySCV49):

    def _public_reference(self, observation):
        value = observation.controller_state.reference_tcp_pose
        p, q = (value.position, value.orientation)
        return matrix_from_xyz_xyzw([p.x, p.y, p.z, q.x, q.y, q.z, q.w])

    def _transfer_approach_gains(self, old, new, *, owner):
        observation = self.gain_get_observation()
        if observation is None:
            raise ValueError('Public observation required for gain transfer')
        state = self.local.assembler.assemble_ros(observation)
        measured = matrix_from_xyz_xyzw(state[:7])
        reference = self._public_reference(observation)
        adjusted = transfer_hold(measured, reference, old, new)
        stamp = self.local._control_sim_time(observation)
        self.local.cartesian_stiffness = list(new)
        self._command(self.gain_move_robot, adjusted, owner=owner)
        self.local.governor_seed = (adjusted.copy(), stamp)
        self._trace(event='free_approach_impedance_gain_transfer', owner=owner, sim_time=stamp, old_stiffness=list(old), new_stiffness=list(new), old_public_reference=reference.tolist(), new_issued_reference=adjusted.tolist(), static_effort_preservation_is_model_based=True, damping_unchanged=True, geometry_input=False)

    def _approach_tcp_rotation(self, port):
        rotation = super()._approach_tcp_rotation(port)
        if self.free_gain_enabled:
            self.approach_old_gains = list(self.local.cartesian_stiffness)
            new = [500.0, 500.0, 240.0, 100.0, 100.0, 100.0]
            self._transfer_approach_gains(self.approach_old_gains, new, owner='free_approach_gain_entry')
            self.approach_active_gains = new
            self.approach_bias_from_issued_reference = True
        return rotation

    def _lock(self, task_vector, get_observation, move_robot, *, fresh_context):
        self.gain_get_observation, self.gain_move_robot = (get_observation, move_robot)
        self.free_gain_enabled = bool(task_vector[0] == 1 or getattr(self.rule, 'direct_empty_selected', False))
        self.approach_old_gains, self.approach_active_gains = (None, None)
        self.approach_bias_from_issued_reference = False
        try:
            return super()._lock(task_vector, get_observation, move_robot, fresh_context=fresh_context)
        finally:
            try:
                if self.approach_active_gains is not None:
                    self._transfer_approach_gains(self.approach_active_gains, self.approach_old_gains, owner='free_approach_gain_exit')
            finally:
                self.gain_get_observation, self.gain_move_robot = (None, None)
                self.approach_bias_from_issued_reference = False

class _RunVisualConnectorServoPrewarmedFreeV51(_RunVisualConnectorServoFreeApproachV50):

    def __init__(self, parent_node):
        super().__init__(parent_node)
        with torch.inference_mode():
            for family, folder in (('sfp', 'sfp_direct_pose'), ('sc', 'sc_orientation_jitter_v2')):
                model = NativeConnectorPoseObservation(ROOT / folder / 'best.pt', str(self.local.device))
                for _ in range(3):
                    model.model(torch.zeros(1, 3, 3, 192, 192, device=self.local.device, dtype=torch.uint8))
                self.connector_models[family] = model
            if self.local.device.type == 'cuda':
                torch.cuda.synchronize(self.local.device)
        self._trace(event='both_grasp_rotation_models_preloaded', model_weights_unchanged=True, synthetic_kernel_warmup_sets_no_pose=True, shared_JPEG_available_before_first_family_query=True, geometry_input=False)

class _RunVisualConnectorServoResponsiveV52(_RunVisualConnectorServoPrewarmedFreeV51):
    parallel_fresh_JPEG = True

    def _make_cached_transport(self, parent_node):
        transport = super()._make_cached_transport(parent_node)
        transport.initial_reject_limit = 1
        return transport

    def _warm_grasp(self):
        super()._warm_grasp()
        self.grasp_jpeg_cache.parallel = True

    def _initial_force_reference(self, get_observation, move_robot):
        original = list(self.local.cartesian_stiffness)
        self.local.cartesian_stiffness = [500.0, 500.0, 500.0, 50.0, 50.0, 50.0]
        self._trace(event='initial_free_hold_stiffness_diagnostic', stiffness_base=self.local.cartesian_stiffness, force_samples_and_spread_gate_unchanged=True, no_contact_retare=True, geometry_input=False)
        try:
            return super()._initial_force_reference(get_observation, move_robot)
        finally:
            self.local.cartesian_stiffness = original

class _RunVisualConnectorServoUniqueForceV53(_RunVisualConnectorServoResponsiveV52):
    unique_initial_force_samples = True
    initial_force_minimum_quiet_span_s = 0.9

    def _close_pose_servo(self, task, get_observation, move_robot, send_feedback, **kwargs):
        if self.local.task_vector(task)[1] == 1 and getattr(self.rule, 'direct_empty_selected', False):
            self.insertion_axial_speed_m_s = 0.024
            return _RunVisualConnectorServoBiasSeedV31._close_pose_servo(self, task, get_observation, move_robot, send_feedback, **kwargs)
        return super()._close_pose_servo(task, get_observation, move_robot, send_feedback, **kwargs)

class _RunVisualConnectorServoPacketTimeV54(_RunVisualConnectorServoUniqueForceV53):
    use_packet_camera_timing = True

    def _make_cached_transport(self, parent_node):
        transport = super()._make_cached_transport(parent_node)
        transport.use_packet_camera_timing = True
        return transport

class _RunVisualConnectorServoJointAcquisitionV55(_RunVisualConnectorServoPacketTimeV54):
    joint_grasp_acquisition = True
    seating_dwell_s = 1.5

    def _make_cached_transport(self, parent_node):
        transport = super()._make_cached_transport(parent_node)
        transport.parallel_transport_JPEG = True
        return transport

class FinalInsertionPolicy(_RunVisualConnectorServoJointAcquisitionV55):
    required_SC_far_context_frames = 3

class _RuntimeBudgetReached(Exception):
    pass

class _TransportLocalContact(Exception):
    pass

class _TransportRecoveryBudget(Exception):
    pass
