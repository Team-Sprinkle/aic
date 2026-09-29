"""Autonomous shared SFP/SC visual BC actor with full port-frame TCP targets.

The checkpoint's pose head supplies the port relation; no TF, scene YAML,
CheatCode, or evaluator geometry is read at runtime. One actor supplies both
normal four-command chunks and, later, recovery decisions after supervisor
control returns. This initial runner has no recovery supervisor or RL update.
"""

from __future__ import annotations

import os
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
from shared_port_tcp_geometry import (inferred_base_port, body_delta_from_target,
                                      matrix_from_xyz_xyzw, matrix_from_pose9)  # noqa: E402
from lerobot_robot_aic.runtime_features import AICRuntimeFeatureAssembler  # noqa: E402


class RunSharedPortTCP(Policy):
    def __init__(self, parent_node, checkpoint_path=None):
        super().__init__(parent_node)
        checkpoint_path = Path(checkpoint_path or os.environ['AIC_SHARED_PORT_TCP_CHECKPOINT'])
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if (checkpoint.get('oracle_sc_near_pose_noise_mm') is not None or
                checkpoint.get('oracle_sc_near_tip_noise_mm') is not None):
            raise ValueError('This diagnostic checkpoint requires an observation-only SC pose input that this runtime does not yet provide')
        if checkpoint.get('architecture') not in (
                'shared_port_tcp_rgb_resnet18_pose_aux_transformer_v1',
                'shared_port_tcp_rgb_resnet18_pose_aux_transformer_time_v2'):
            raise ValueError('Incompatible shared port-TCP checkpoint')
        self.device = torch.device(os.environ.get('AIC_SHARED_PORT_TCP_DEVICE', 'cuda:0'))
        state = checkpoint['model']
        stats = {key: state[key].numpy() for key in ('state_mean', 'state_std',
                 'pose_mean', 'pose_std', 'target_mean', 'target_std')}
        self.include_elapsed = bool(checkpoint.get('include_elapsed', False))
        self.target_frame = checkpoint.get('target_frame', 'port')
        if self.target_frame not in ('port', 'body'):
            raise ValueError(f'Unsupported shared actor target frame: {self.target_frame}')
        self.time_coordinate = checkpoint.get('time_coordinate', 'sim_elapsed')
        if self.time_coordinate not in ('sim_elapsed', 'nominal_decision'):
            raise ValueError(f'Unsupported shared-actor time coordinate: {self.time_coordinate}')
        self.actor = SharedPortTCPActor(stats, pretrained=False,
                                       use_time=self.include_elapsed,
                                       condition_on_pose=self.target_frame == 'port')
        self.actor.load_state_dict(state, strict=True)
        self.actor.to(self.device).eval()
        self.assembler = AICRuntimeFeatureAssembler(32, fps=20.)
        self.control_hz = float(os.environ.get('AIC_SHARED_PORT_TCP_CONTROL_HZ', '20'))
        self.max_runtime_sec = float(os.environ.get('AIC_SHARED_PORT_TCP_MAX_RUNTIME_SEC', '90'))
        self.max_translation = float(os.environ.get('AIC_SHARED_PORT_TCP_MAX_TRANSLATION_DELTA', '.02'))
        self.max_rotation = float(os.environ.get('AIC_SHARED_PORT_TCP_MAX_ROTATION_DELTA', '.2'))
        self.replan_every_command = os.environ.get('AIC_SHARED_PORT_TCP_REPLAN_EVERY_COMMAND', '0') == '1'
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
    def camera_tensor(message):
        enc = message.encoding.lower()
        channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4}.get(enc)
        if channels is None:
            raise ValueError(f'Unsupported camera encoding {message.encoding}')
        raw = np.frombuffer(message.data, np.uint8).reshape(message.height, message.step)
        rgb = raw[:, :message.width*channels].reshape(message.height, message.width, channels)[..., :3]
        if enc.startswith('bgr'):
            rgb = rgb[..., ::-1]
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
        images = torch.stack([self.camera_tensor(getattr(observation, name + '_image'))
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
        self.assembler.reset()
        self.first_sim_time = None
        self.last_elapsed_sim_sec = 0.
        self.student_proposal_count = 0
        self._inference_latency_ms.clear()
        self._student_task_vector = self.task_vector(task)

    def clear_pending_actions(self):
        # This actor predicts a fresh chunk on each observation, with no
        # queued future command to leak across a teacher-controlled segment.
        return None

    def _target_anchor(self, tcp_base, estimated_pose):
        return (inferred_base_port(tcp_base, estimated_pose) if self.target_frame == 'port'
                else matrix_from_xyz_xyzw(tcp_base))

    def _body_delta(self, tcp_base, anchor, target):
        if self.target_frame == 'port':
            return body_delta_from_target(tcp_base, anchor, target)
        relative = np.linalg.inv(matrix_from_xyz_xyzw(tcp_base)) @ anchor @ matrix_from_pose9(target)
        return np.concatenate((relative[:3, 3],
                               Rotation.from_matrix(relative[:3, :3]).as_rotvec()))

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
        self.assembler.reset()
        self.first_sim_time = None
        self.last_elapsed_sim_sec = 0.
        self.student_proposal_count = 0
        self._inference_latency_ms.clear()
        task_vector = self.task_vector(task)
        start = time.monotonic()
        issued = 0
        period = 1 / self.control_hz
        while time.monotonic() - start < self.max_runtime_sec:
            observed = get_observation()
            if observed is None:
                time.sleep(period)
                continue
            targets, estimated_pose, anchor_state = self.predict_chunk(
                observed, task_vector, decision_time_s=issued / self.control_hz)
            anchor = self._target_anchor(anchor_state[:7], estimated_pose)
            for index, target in enumerate(targets[:1] if self.replan_every_command else targets):
                if time.monotonic() - start >= self.max_runtime_sec:
                    break
                tick = time.monotonic()
                if index == 0:
                    state = anchor_state
                else:
                    current = get_observation()
                    if current is None:
                        break
                    state = self.assembler.assemble_ros(current)
                delta = (np.zeros(6) if self.last_elapsed_sim_sec < self.startup_hold_sim_sec
                         else self._body_delta(state[:7], anchor, target))
                if not np.isfinite(delta).all():
                    raise ValueError('Nonfinite predicted port-frame TCP command')
                force_norm = float(np.linalg.norm(state[26:29]))
                if force_norm > self.diagnostic_force_stop_n:
                    self.get_logger().error(f'Shared BC diagnostic force stop: {force_norm:.1f} N')
                    self._report_latency()
                    return False
                translation, rotation = clamp_delta_pose_components(
                    delta[:3], delta[3:], max_translation=self.max_translation,
                    max_rotation=self.max_rotation)
                body_target = np.eye(4)
                body_target[:3, 3] = translation
                body_target[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
                absolute = matrix_from_xyz_xyzw(state[:7]) @ body_target
                pose = build_pose_from_vectors(absolute[:3, 3],
                                               Rotation.from_matrix(absolute[:3, :3]).as_quat())
                if hasattr(self._parent_node, '_target_mode'):
                    self._parent_node._target_mode = TargetMode.MODE_CARTESIAN
                self.set_pose_target(move_robot=move_robot, pose=pose, frame_id='base_link')
                issued += 1
                if issued % max(1, round(self.control_hz)) == 0:
                    send_feedback(f'in progress; shared port TCP commands={issued}')
                time.sleep(max(0., period - (time.monotonic() - tick)))
        self._report_latency()
        return True

    def _report_latency(self):
        if self._inference_latency_ms:
            values = np.asarray(self._inference_latency_ms)
            p50, p95, p99 = np.quantile(values, [.5, .95, .99])
            self.get_logger().info(
                f'Shared BC camera/state/model inference latency ms: '
                f'n={len(values)} p50={p50:.2f} p95={p95:.2f} p99={p99:.2f}')
