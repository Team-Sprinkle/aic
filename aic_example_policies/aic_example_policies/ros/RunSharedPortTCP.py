"""Autonomous shared SFP/SC visual BC actor with full port-frame TCP targets.

The checkpoint's pose head supplies the port relation; no TF, scene YAML,
CheatCode, or evaluator geometry is read at runtime. One actor supplies both
normal four-command chunks and, later, recovery decisions after supervisor
control returns. This initial runner has no recovery supervisor or RL update.
"""

from __future__ import annotations

import os
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from aic_control_interfaces.msg import TargetMode
from aic_model.policy import Policy, build_pose_from_vectors, clamp_delta_pose_components
from aic_task_interfaces.msg import Task
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / 'scripts'))
sys.path.insert(0, str(REPO_ROOT / 'aic_utils' / 'lerobot_robot_aic'))
from train_shared_port_tcp_bc import SharedPortTCPActor  # noqa: E402
from shared_bc_camera_preprocessing import transport_rgb_tensor  # noqa: E402
from shared_port_tcp_geometry import (inferred_base_port, body_delta_from_target,
                                      matrix_from_xyz_xyzw, matrix_from_pose9)  # noqa: E402
from lerobot_robot_aic.runtime_features import AICRuntimeFeatureAssembler  # noqa: E402


class RunSharedPortTCP(Policy):
    supports_sc_tip_target_reference=False

    @staticmethod
    def _make_actor(checkpoint, stats, **kwargs):
        return SharedPortTCPActor(stats, **kwargs)
    def __init__(self, parent_node, checkpoint_path=None):
        super().__init__(parent_node)
        checkpoint_path = Path(checkpoint_path or os.environ['AIC_SHARED_PORT_TCP_CHECKPOINT'])
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        # Old locked-port checkpoints retain their SC-only runtime contract.
        self.trained_families=tuple(checkpoint.get('trained_families',('sc',)))
        self.recorded_controller_contract=checkpoint.get('recorded_controller_contract')
        if (checkpoint.get('oracle_sc_near_pose_noise_mm') is not None or
                checkpoint.get('oracle_sc_near_tip_noise_mm') is not None):
            raise ValueError('This diagnostic checkpoint requires an observation-only SC pose input that this runtime does not yet provide')
        if checkpoint.get('architecture') not in (
                'shared_port_tcp_rgb_resnet18_pose_aux_transformer_v1',
                'shared_port_tcp_rgb_resnet18_pose_aux_transformer_time_v2',
                'shared_port_tcp_local_predicted_pose'):
            raise ValueError('Incompatible shared port-TCP checkpoint')
        self.device = torch.device(os.environ.get('AIC_SHARED_PORT_TCP_DEVICE', 'cuda:0'))
        state = checkpoint['model']
        stats = {key: state[key].numpy() for key in ('state_mean', 'state_std',
                 'pose_mean', 'pose_std', 'target_mean', 'target_std')}
        self.include_tip=bool(checkpoint.get('include_tip',False))
        self.tip_input_mode=checkpoint.get('tip_input_mode','none')
        self.sc_action_reference=checkpoint.get('sc_action_reference','tcp')
        if self.sc_action_reference not in ('tcp','plug_tip_translation_tcp_orientation','full_connector_pose'):
            raise ValueError('Unsupported SC action reference')
        if self.sc_action_reference!='tcp' and not self.supports_sc_tip_target_reference:
            raise ValueError('This runtime cannot convert plug-tip targets; use the explicitly matched locked-port adapter')
        if self.sc_action_reference!='tcp' and self.tip_input_mode not in ('translation','full_pose'):
            raise ValueError('Point targets require camera-estimated body-tip conversion')
        if self.include_tip:
            if self.tip_input_mode not in ('disabled','constant_translation','translation','full_pose'):
                raise ValueError('Unspecified predicted-tip runtime contract')
            stats.update({key:state[key].numpy() for key in ('tip_mean','tip_std')})
        self.include_elapsed = bool(checkpoint.get('include_elapsed', False))
        self.target_frame = checkpoint.get('target_frame', 'port')
        self.native_jpeg_quality_by_family = checkpoint.get('native_jpeg_quality_by_family', {})
        if self.target_frame not in ('port', 'estimated_port', 'body'):
            raise ValueError(f'Unsupported shared actor target frame: {self.target_frame}')
        self.time_coordinate = checkpoint.get('time_coordinate', 'sim_elapsed')
        if self.time_coordinate not in ('sim_elapsed', 'nominal_decision'):
            raise ValueError(f'Unsupported shared-actor time coordinate: {self.time_coordinate}')
        self.actor = self._make_actor(checkpoint, stats, pretrained=False,
                                       use_time=self.include_elapsed,
                                       use_tip=self.include_tip,
                                       condition_on_pose=self.target_frame in ('port','estimated_port'),
                                       separate_family_orientation_head=bool(
                                           checkpoint.get('separate_family_orientation_head',False)),
                                       separate_family_action_head=bool(
                                           checkpoint.get('separate_family_action_head',False)),
                                       use_local_history=bool(checkpoint.get('use_local_history',False)),
                                       decoder_kind=checkpoint.get('decoder_kind','transformer'))
        self.actor.load_state_dict(state, strict=True)
        masked=checkpoint.get('masked_state_indices',[])
        if any(not isinstance(i,int) or not 0<=i<32 for i in masked):
            raise ValueError('Invalid checkpoint state input mask')
        self.actor.masked_state_indices=masked
        self.actor.masked_tip_indices=checkpoint.get('masked_tip_indices',[])
        self.actor.tip_input_mode=self.tip_input_mode
        self.actor.mask_history_command_feedback=bool(checkpoint.get('mask_history_command_feedback',False))
        self.actor.coordinate_heads_fp32=bool(checkpoint.get('coordinate_heads_fp32',False))
        self.actor.control_path_fp32=bool(checkpoint.get('control_path_fp32',False))
        self.constant_body_tip=checkpoint.get('constant_body_tip_mean')
        self.history_offsets_s=checkpoint.get('history_offsets_s')
        self.history_masked=bool(checkpoint.get('history_masked',False))
        self.actor.history_max_age_s=max(self.history_offsets_s) if self.history_offsets_s else 2.
        self.actor.to(self.device).eval()
        self.assembler = AICRuntimeFeatureAssembler(32, fps=20.)
        self.control_hz = float(os.environ.get('AIC_SHARED_PORT_TCP_CONTROL_HZ', '20'))
        self.control_clock = os.environ.get('AIC_SHARED_PORT_TCP_CONTROL_CLOCK', 'wall')
        if self.control_clock not in ('wall', 'simulation'):
            raise ValueError('Unknown shared actor control clock')
        self.max_simulation_sec = float(os.environ.get('AIC_SHARED_PORT_TCP_MAX_SIMULATION_SEC', '70'))
        self.trace_path = os.environ.get('AIC_SHARED_PORT_TCP_TRACE', '')
        self.cartesian_stiffness = json.loads(os.environ.get(
            'AIC_SHARED_PORT_TCP_CARTESIAN_STIFFNESS', '[90,90,90,50,50,50]'))
        self.cartesian_damping = json.loads(os.environ.get(
            'AIC_SHARED_PORT_TCP_CARTESIAN_DAMPING', '[50,50,50,20,20,20]'))
        if any(len(v)!=6 or not all(np.isfinite(x) and x>0 for x in v)
               for v in (self.cartesian_stiffness,self.cartesian_damping)):
            raise ValueError('Invalid Cartesian gain vector')
        self.max_runtime_sec = float(os.environ.get('AIC_SHARED_PORT_TCP_MAX_RUNTIME_SEC', '90'))
        self.max_translation = float(os.environ.get('AIC_SHARED_PORT_TCP_MAX_TRANSLATION_DELTA', '.02'))
        self.max_rotation = float(os.environ.get('AIC_SHARED_PORT_TCP_MAX_ROTATION_DELTA', '.2'))
        self.replan_every_command = os.environ.get('AIC_SHARED_PORT_TCP_REPLAN_EVERY_COMMAND', '0') == '1'
        self.execute_steps = None
        self._action_generation = 0
        self.execute_complete_chunk = os.environ.get('AIC_SHARED_PORT_TCP_EXECUTE_COMPLETE_CHUNK','0')=='1'
        if self.execute_complete_chunk and (self.control_clock!='simulation' or self.replan_every_command):
            raise ValueError('Complete-chunk execution requires simulation clock and chunk control')
        self.diagnostic_force_stop_n = float(os.environ.get('AIC_SHARED_PORT_TCP_DIAGNOSTIC_FORCE_STOP_N', '40'))
        self.startup_hold_sim_sec = float(os.environ.get('AIC_SHARED_PORT_TCP_STARTUP_HOLD_SIM_SEC', '0'))
        if self.startup_hold_sim_sec < 0:
            raise ValueError('Startup hold duration must be nonnegative')
        self.first_sim_time = None
        self.last_elapsed_sim_sec = 0.
        self.student_proposal_count = 0
        self._student_task_vector = None
        self._inference_latency_ms = []
        self.get_logger().info(f'Shared port-TCP BC loaded from {checkpoint_path}; '
                               f'update={checkpoint["update"]}, device={self.device}')

    @staticmethod
    def task_vector(task):
        if task.target_module_name.startswith('nic_card_mount_'):
            card = int(task.target_module_name.rsplit('_', 1)[-1])
            port = int(task.port_name.rsplit('_', 1)[-1])
            if card not in range(5) or port not in (0, 1):
                raise ValueError('Invalid SFP task encoding')
            one_hot = [0.] * 5
            one_hot[card] = 1.
            return [1., 0., float(port == 0), float(port == 1), *one_hot, 1.]
        if task.target_module_name.startswith('sc_port_'):
            port = int(task.target_module_name.rsplit('_', 1)[-1])
            if port not in (0, 1):
                raise ValueError('Invalid SC target port')
            return [0., 1., float(port == 0), float(port == 1), 0., 0., 0., 0., 0., 0.]
        raise ValueError('Unknown AIC target module')

    @staticmethod
    def camera_tensor(message, native_jpeg_quality=None):
        enc = message.encoding.lower()
        channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4}.get(enc)
        if channels is None:
            raise ValueError(f'Unsupported camera encoding {message.encoding}')
        raw = np.frombuffer(message.data, np.uint8).reshape(message.height, message.step)
        rgb = raw[:, :message.width*channels].reshape(message.height, message.width, channels)[..., :3]
        if enc.startswith('bgr'):
            rgb = rgb[..., ::-1]
        if native_jpeg_quality is not None:
            return transport_rgb_tensor(rgb, native_jpeg_quality=native_jpeg_quality)
        # Match supervised Samples: collector saves 288x256 RGB as quality-95
        # JPEG, then PIL resizes that decoded image to 224x224 with BILINEAR.
        # Direct native->224 INTER_AREA changed target predictions by millimeters.
        intermediate = cv2.resize(rgb, (288, 256), interpolation=cv2.INTER_AREA)
        success, encoded = cv2.imencode('.jpg', intermediate[..., ::-1],
                                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            raise RuntimeError('Failed to encode shared BC camera image')
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError('Failed to decode shared BC camera image')
        resized = np.asarray(Image.fromarray(decoded[..., ::-1]).resize(
            (224, 224), Image.Resampling.BILINEAR))
        return torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1)

    def predict_chunk(self, observation, task_vector, *, decision_time_s=None):
        inference_start = time.perf_counter()
        family = 'sfp' if task_vector[0] == 1 else 'sc'
        native_quality = self.native_jpeg_quality_by_family.get(family)
        images = torch.stack([self.camera_tensor(getattr(observation, name + '_image'), native_quality)
                              for name in ('center', 'left', 'right')])[None].to(self.device)
        state = self.assembler.assemble_ros(observation)
        if state.shape != (32,):
            raise ValueError(f'Expected 32D ordinary robot state, got {state.shape}')
        stamp = observation.center_image.header.stamp
        sim_time = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        if self.first_sim_time is None:
            self.first_sim_time = sim_time
        self.last_elapsed_sim_sec = max(0., sim_time - self.first_sim_time)
        if self.time_coordinate == 'nominal_decision':
            if decision_time_s is None:
                raise ValueError('Nominal-decision checkpoint requires command-clock time')
            elapsed = float(decision_time_s)
        else:
            elapsed = self.last_elapsed_sim_sec
        with torch.inference_mode():
            target, pose = self.actor(images,
                torch.as_tensor(state, device=self.device)[None],
                torch.as_tensor(task_vector, dtype=torch.float32, device=self.device)[None],
                torch.tensor([elapsed], dtype=torch.float32, device=self.device))
        output = target[0].cpu().numpy(), pose[0].cpu().numpy(), state
        self._inference_latency_ms.append((time.perf_counter()-inference_start)*1000)
        return output

    def reset_for_task(self, task):
        """Reset causal state for bounded teacher-collection interventions."""
        self.clear_pending_actions()
        self.assembler.reset()
        self.first_sim_time = None
        self.last_elapsed_sim_sec = 0.
        self.student_proposal_count = 0
        self._inference_latency_ms.clear()
        self._student_task_vector = self.task_vector(task)

    def clear_pending_actions(self):
        # A chunk is retained by insert_cable's command loop. Ownership
        # changes must invalidate that local chunk as well as cached plans.
        self._action_generation += 1

    def _target_anchor(self, tcp_base, estimated_pose):
        return (inferred_base_port(tcp_base, estimated_pose) if self.target_frame in ('port','estimated_port')
                else matrix_from_xyz_xyzw(tcp_base))

    def _body_delta(self, tcp_base, anchor, target):
        if self.target_frame in ('port','estimated_port'):
            return body_delta_from_target(tcp_base, anchor, target)
        relative = np.linalg.inv(matrix_from_xyz_xyzw(tcp_base)) @ anchor @ matrix_from_pose9(target)
        return np.concatenate((relative[:3, 3],
                               Rotation.from_matrix(relative[:3, :3]).as_rotvec()))

    def _should_stop_transport(self, task_vector):
        """Observation-only optional handoff hook for transport-only diagnostics."""
        return False

    def _inspect_transport_handoff(self, observed, state, task_vector,
                                   get_observation, move_robot, decision_sim):
        """Return whether a proposed stop is accepted; subclasses may inspect."""
        return True

    def _cartesian_gains(self):
        return self.cartesian_stiffness, self.cartesian_damping

    def _bound_transport_target(self, absolute, task_vector, state):
        return absolute

    def select_delta_action(self, observation, *, nominal_time_s=None):
        """Return an absolute base pose as xyz plus world rotvec.

        The privileged collector clips this *proposal* around its teacher
        target and records both. This method itself uses only observations.
        """
        if self._student_task_vector is None:
            raise RuntimeError('reset_for_task must precede student inference')
        targets, estimated_pose, state = self.predict_chunk(
            observation, self._student_task_vector,
            decision_time_s=(self.student_proposal_count / self.control_hz
                             if nominal_time_s is None else nominal_time_s))
        self.student_proposal_count += 1
        anchor = self._target_anchor(state[:7], estimated_pose)
        delta = self._body_delta(state[:7], anchor, targets[0])
        translation, rotation = clamp_delta_pose_components(
            delta[:3], delta[3:], max_translation=self.max_translation,
            max_rotation=self.max_rotation)
        body_target = np.eye(4)
        body_target[:3, 3] = translation
        body_target[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
        absolute = matrix_from_xyz_xyzw(state[:7]) @ body_target
        return np.concatenate((absolute[:3, 3],
                               Rotation.from_matrix(absolute[:3, :3]).as_rotvec()))

    def insert_cable(self, task: Task, get_observation, move_robot, send_feedback, **kwargs) -> bool:
        # Dispatch subclass reset as well: route stages and visual memories
        # must never survive a new episode/action request.
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
                    time.sleep(.002)
                    continue
            plan_generation = self._action_generation
            targets, estimated_pose, anchor_state = self.predict_chunk(
                observed, task_vector, decision_time_s=issued / self.control_hz)
            self._trace(event='proposal', sim_time=decision_sim, state=anchor_state.tolist(),
                        targets=targets.tolist(), task_vector=task_vector,
                        masked_state_indices=self.actor.masked_state_indices,
                        image_sim_ns={name:getattr(observed,name+'_image').header.stamp.sec*1000000000+
                                      getattr(observed,name+'_image').header.stamp.nanosec
                                      for name in ('center','left','right')},
                        inference_ms=self._inference_latency_ms[-1], geometry_input=False)
            # A rejected inspection must permit a fresh motion chunk before
            # another inspection. Otherwise a high proposal score can trap
            # the policy in repeated holds without any corrective motion.
            should_inspect = not resume_after_rejection and self._should_stop_transport(task_vector)
            resume_after_rejection = False
            if should_inspect:
                self._handoff_terminal_snapshot = None
                if not self._inspect_transport_handoff(observed, anchor_state,
                        task_vector, get_observation, move_robot, decision_sim):
                    # A shadow inspection commanded a hold and rejected the
                    # handoff. Resume normal transport from a fresh clock tick.
                    next_decision_sim = decision_sim + period
                    resume_after_rejection = True
                    if hasattr(self, 'stop_streak'):
                        self.stop_streak = 0
                    continue
                # Inspectors can spend seconds holding before approval. Retain
                # their accepted observation; never fetch geometry after reset
                # or mislabel the earlier proposal as the terminal transition.
                snapshot = getattr(self, '_handoff_terminal_snapshot', None)
                terminal_state, terminal_sim = (snapshot[:2] if snapshot is not None
                                                else (anchor_state, decision_sim))
                self._trace(event='terminal_before_return', reason=getattr(self,'transport_terminal_reason','learned_transport_handoff'),
                            state=terminal_state.tolist(), sim_time=terminal_sim,
                            observation_source=('accepted_handoff_observation' if snapshot is not None
                                                else 'current_decision_observation'),
                            geometry_input=False)
                self._report_latency()
                return False
            anchor = self._target_anchor(anchor_state[:7], estimated_pose)
            planned = targets[:1] if self.replan_every_command else targets[:self.execute_steps]
            next_scheduled_sim=None
            if self.execute_complete_chunk:
                # Inference can take longer than the chunk. Start its first
                # command at the fresh post-inference observation and space
                # subsequent commands by measured simulation time.
                fresh=get_observation()
                if fresh is None:break
                next_scheduled_sim=max(decision_sim,self._control_sim_time(fresh))
            for index, target in enumerate(planned):
                if plan_generation != self._action_generation:
                    self._trace(event='chunk_cancelled_for_control_transfer',index=index)
                    break
                if time.monotonic() - start >= self.max_runtime_sec:
                    break
                tick = time.monotonic()
                scheduled_sim = (next_scheduled_sim if self.execute_complete_chunk
                                 else decision_sim + index * period)
                if self.control_clock == 'simulation':
                    current = get_observation()
                    while (current is not None and self._control_sim_time(current) < scheduled_sim
                           and time.monotonic() - start < self.max_runtime_sec):
                        time.sleep(.002)
                        current = get_observation()
                    if current is None:
                        break
                    if (not self.execute_complete_chunk and index > 0 and
                            self._control_sim_time(current) > scheduled_sim + period):
                        self._trace(event='stale_chunk_member_dropped', index=index,
                                    sim_time=self._control_sim_time(current), scheduled_sim=scheduled_sim)
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
                    self._trace(event='chunk_cancelled_for_control_transfer',index=index)
                    break
                delta = (np.zeros(6) if self.last_elapsed_sim_sec < self.startup_hold_sim_sec
                         else self._body_delta(state[:7], anchor, target))
                if not np.isfinite(delta).all():
                    raise ValueError('Nonfinite predicted port-frame TCP command')
                force_norm = float(np.linalg.norm(state[26:29]))
                if force_norm > self.diagnostic_force_stop_n:
                    self.get_logger().error(f'Shared BC diagnostic force stop: {force_norm:.1f} N')
                    retained_observation=(current if self.control_clock=='simulation' or index>0 else observed)
                    self._trace(event='terminal_before_return', reason='force_stop',
                                sim_time=self._control_sim_time(retained_observation),
                                state=state.tolist(), force_norm_n=force_norm,
                                observation_source='current_command_tick',post_reset_geometry_used=False)
                    self._report_latency()
                    return False
                translation, rotation = clamp_delta_pose_components(
                    delta[:3], delta[3:], max_translation=self.max_translation,
                    max_rotation=self.max_rotation)
                body_target = np.eye(4)
                body_target[:3, 3] = translation
                body_target[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
                absolute = matrix_from_xyz_xyzw(state[:7]) @ body_target
                absolute = self._bound_transport_target(absolute, task_vector, state)
                effective_body = np.linalg.inv(matrix_from_xyz_xyzw(state[:7])) @ absolute
                effective_delta = np.r_[effective_body[:3, 3],
                    Rotation.from_matrix(effective_body[:3, :3]).as_rotvec()]
                pose = build_pose_from_vectors(absolute[:3, 3],
                                               Rotation.from_matrix(absolute[:3, :3]).as_quat())
                if hasattr(self._parent_node, '_target_mode'):
                    self._parent_node._target_mode = TargetMode.MODE_CARTESIAN
                stiffness,damping=self._cartesian_gains()
                if plan_generation != self._action_generation:
                    self._trace(event='chunk_cancelled_for_control_transfer',index=index)
                    break
                self.set_pose_target(move_robot=move_robot, pose=pose, frame_id='base_link',
                                     stiffness=stiffness,damping=damping)
                self._trace(event='command', sim_time=(self._control_sim_time(current)
                            if self.control_clock == 'simulation' else decision_sim),
                            scheduled_sim=scheduled_sim, chunk_index=index,
                            state=state.tolist(), delta=effective_delta.tolist(),
                            absolute_tcp_target=absolute.tolist(),
                            stiffness=stiffness,damping=damping)
                if self.execute_complete_chunk:
                    next_scheduled_sim=self._control_sim_time(current)+period
                issued += 1
                if issued % max(1, round(self.control_hz)) == 0:
                    send_feedback(f'in progress; shared port TCP commands={issued}')
                if self.control_clock == 'wall':
                    time.sleep(max(0., period - (time.monotonic() - tick)))
            next_decision_sim = (next_scheduled_sim if self.execute_complete_chunk
                                 else decision_sim + len(planned)*period)
        terminal = get_observation()
        if terminal is not None:
            self._trace(event='terminal_before_return', reason='bounded_runtime',
                        sim_time=self._control_sim_time(terminal),
                        state=self.assembler.assemble_ros(terminal).tolist())
        self._report_latency()
        # This transport-only runner has no insertion detector. A time limit
        # cannot be reported as a successful insert_cable action.
        return False

    @staticmethod
    def _control_sim_time(observation):
        stamp = observation.controller_state.header.stamp
        return float(stamp.sec) + float(stamp.nanosec)*1e-9

    def _trace(self, **record):
        if self.trace_path:
            with open(self.trace_path, 'a') as handle:
                handle.write(json.dumps(record)+'\n')

    def _report_latency(self):
        if self._inference_latency_ms:
            values = np.asarray(self._inference_latency_ms)
            p50, p95, p99 = np.quantile(values, [.5, .95, .99])
            self.get_logger().info(
                f'Shared BC camera/state/model inference latency ms: '
                f'n={len(values)} p50={p50:.2f} p95={p95:.2f} p99={p99:.2f}')
