"""Image-only full SC grasp pose from the corrected spatial fit schema."""
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from train_sc_native_keypoints import PretrainedKeypointRefiner
from sc_native_grasp_observation import SCNativeGraspObservation


class SCSpatialGraspObservation(SCNativeGraspObservation):
    def __init__(self, path, device='cuda', *, approval=None):
        path = Path(path)
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['schema'] != 'sc_rgb_grasp_geometry/v1' or saved['label_time_contract'] != 'causal_image_time_and_same_time_TCP':
            raise ValueError('Correct image-time spatial grasp schema required')
        if not saved['plan']['geometry_labels_only']:
            raise ValueError('Simulator geometry must remain supervision only')
        self.approved = False
        if approval is not None:
            report = json.loads(Path(approval).read_text())
            if (not report['pose_gate_passed'] or not report.get('visual_inspection_completed', False) or
                    report.get('pose_gate_scope') != 'SC_held_near25mm_only' or
                    report['checkpoint_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest()):
                raise ValueError('Exact-checkpoint held-out pose approval required')
            self.approved = True
        self.device = torch.device(device)
        self.cameras = ('center', 'left', 'right')
        self.centres = saved['crop_centres']
        self.optics = {c: np.asarray(saved['tcp_to_optical'][c]) for c in self.cameras}
        self.cad = np.asarray(saved['CAD_reference_points'])
        self.native_sizes = saved.get('native_crop_sizes')
        self.native_origins = saved.get('native_crop_origins')
        if (self.native_sizes is None) != (self.native_origins is None):
            raise ValueError('Compact crop sizes and native origins must travel together')
        self.model = PretrainedKeypointRefiner(None, False, point_count=5, condition_camera=True)
        self.model.backbone.layer4 = nn.Identity()
        self.model.backbone.avgpool = nn.Identity()
        self.model.backbone.fc = nn.Identity()
        self.model.load_state_dict(saved['model'], strict=True)
        self.model.to(self.device).eval()
        self.latency_ms = []

    @torch.inference_mode()
    def predict(self, images, *, already_training_jpeg=False):
        if self.native_sizes is None:
            result = super().predict(images, already_training_jpeg=already_training_jpeg)
        else:
            from evaluate_sc_native_triangulation import triangulate
            start = time.perf_counter()
            patches = []
            for camera in self.cameras:
                image = images[camera]
                if image.shape != (1024, 1152, 3):
                    raise ValueError('Original camera pixels are required')
                if not already_training_jpeg:
                    ok, encoded = cv2.imencode('.jpg', image[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 86])
                    if not ok:
                        raise RuntimeError('Native training codec failed')
                    image = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                x, y = self.native_origins[camera]
                scale = 191/(self.native_sizes[camera]-1)
                patches.append(cv2.warpAffine(image, np.array(
                    [[scale, 0, -scale*x], [0, scale, -scale*y]], np.float32), (192, 192)))
            x = torch.from_numpy(np.stack(patches)).permute(0, 3, 1, 2).to(self.device).float()/255
            sizes = np.array([self.native_sizes[c]-1 for c in self.cameras])
            origins = np.array([self.native_origins[c] for c in self.cameras])
            pixels = (self.model(x, camera_id=torch.eye(3, device=self.device)).cpu().numpy().reshape(3, 5, 2)
                      * sizes[:, None, None] + origins[:, None, :])
            cloud = np.stack([triangulate({c: pixels[i, k] for i, c in enumerate(self.cameras)}, self.optics)
                              for k in range(5)])
            a = self.cad-self.cad.mean(0)
            b = cloud-cloud.mean(0)
            u, _, vh = np.linalg.svd(a.T@b)
            rotation = vh.T@np.diag([1, 1, np.linalg.det(vh.T@u.T)])@u.T
            position = cloud.mean(0)-rotation@self.cad.mean(0)
            duration = (time.perf_counter()-start)*1000
            self.latency_ms.append(duration)
            result = dict(predicted_tip_position_tcp_m=position.tolist(), predicted_pixels=pixels.tolist(),
                          orientation_for_diagnostic_only=rotation.tolist(), runtime_true_geometry=False,
                          latency_ms=duration, input='Three fixed train-derived compact native RGB crops; no GT selection')
        rotation = np.asarray(result['orientation_for_diagnostic_only'])
        position = np.asarray(result['predicted_tip_position_tcp_m'])
        cloud = self.cad @ rotation.T + position
        errors = []
        from evaluate_sc_native_triangulation import FOCAL
        for index, camera in enumerate(self.cameras):
            inverse = np.linalg.inv(self.optics[camera])
            points = cloud @ inverse[:3, :3].T + inverse[:3, 3]
            projected = points[:, :2]/points[:, 2:]*FOCAL + [576, 512]
            errors.extend(np.linalg.norm(projected-np.asarray(result['predicted_pixels'])[index], axis=-1))
        result.update(translation_only=False, orientation_approved=self.approved,
                      spatial_pose_schema='sc_rgb_grasp_geometry/v1',
                      rigid_fit_reprojection_max_px=float(max(errors)),
                      rigid_fit_reprojection_rms_px=float(np.sqrt(np.mean(np.square(errors)))))
        return result
