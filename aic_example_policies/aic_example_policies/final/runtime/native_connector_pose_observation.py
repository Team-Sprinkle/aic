"""Inference definitions extracted from native_connector_pose_observation.py; no training entry point."""
import time
import hashlib
from pathlib import Path
import cv2
import numpy as np
import torch
from .models_native_connector_pose import ConnectorPose

class NativeConnectorPoseObservation:

    def __init__(self, checkpoint, device='cuda'):
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.plan = saved['plan']
        if self.plan['schema'] != 'native_connector_full_pose/v1' or not self.plan['geometry_labels_only']:
            raise ValueError('Expected image-only connector pose checkpoint')
        state = saved['model']
        encoder = {k[len('landmark.'):]: v for k, v in state.items() if k.startswith('landmark.')}
        model_class = ConnectorPose
        if self.plan.get('architecture') == 'fine_spatial_orientation_v1':
            source = Path(__file__).with_name('fine_connector_orientation.py')
            if '522b62db23e59b8f2bfc1ba9c9504df9875264ffca3b4ca7c9044b0a7fd76036' != self.plan['fine_model_source_sha256']:
                raise ValueError('Fine-feature inference source differs from training')
            from .fine_connector_orientation import FineConnectorPose
            model_class = FineConnectorPose
        self.model = model_class(encoder, saved['pose_mean'], saved['pose_scale'])
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval()
        self.device = torch.device(device)
        self.origins = np.asarray(self.plan['crop_origins_native'])
        self.rotation_only = bool(self.plan.get('rotation_only'))
        self.latency_ms = []

    @torch.inference_mode()
    def predict(self, images, *, already_training_jpeg=False, visual_tip_position_tcp_m=None):
        started = time.perf_counter()
        patches = []
        for ci, c in enumerate(['center', 'left', 'right']):
            rgb = images[c]
            if rgb.shape != (1024, 1152, 3) or rgb.dtype != np.uint8:
                raise ValueError('Original uint8 RGB images required')
            if not already_training_jpeg:
                ok, encoded = cv2.imencode('.jpg', rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 86])
                if not ok:
                    raise ValueError('Training codec failed')
                rgb = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            x, y = self.origins[ci]
            patches.append(cv2.warpAffine(rgb, np.array([[1, 0, -x], [0, 1, -y]], np.float32), (192, 192)))
        x = torch.from_numpy(np.stack(patches)).permute(0, 3, 1, 2)[None].to(self.device)
        position, rotation = self.model(x)
        position = position.cpu().numpy()[0]
        rotation = rotation.cpu().numpy()[0]
        if self.rotation_only and visual_tip_position_tcp_m is None:
            raise ValueError('Rotation-only checkpoint needs an explicit RGB-derived position estimate')
        if visual_tip_position_tcp_m is not None:
            position = np.asarray(visual_tip_position_tcp_m, dtype=float)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError('Bad visual tip position')
        pose = np.eye(4)
        pose[:3, 3] = position
        pose[:3, :3] = rotation
        latency = (time.perf_counter() - started) * 1000
        self.latency_ms.append(latency)
        return dict(predicted_connector_in_tcp_matrix=pose.tolist(), predicted_tip_position_tcp_m=position.tolist(), orientation_for_diagnostic_only=rotation.tolist(), orientation_approved=False, runtime_true_geometry=False, latency_ms=latency, position_source='existing RGB estimator' if visual_tip_position_tcp_m is not None else 'joint RGB pose head', note='Offline candidate; frozen production actors remain unchanged')
