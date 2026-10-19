"""Observation-only selected SFP port cascade for near-port inference.

The task code selects a candidate identity. A frozen RGB locator chooses a
480-pixel native crop, then a second RGB head chooses a 160-pixel native crop.
Three refined views triangulate the target. Ground-truth geometry is never
read here; camera calibration and measured TCP are the only geometry inputs.
"""
import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from audit_sc_port_targets import pose_matrix
from build_sc_native_pose_labels import project
from evaluate_sc_native_triangulation import triangulate, reprojection_residual_px
from train_sc_native_keypoints import PretrainedKeypointRefiner
from train_two_family_port_locator import CAMERAS, NAMES, Locator
from train_two_family_port_orientation import Model


class SfpPortCascade:
    def __init__(self, locator_path, broad_path, fine_path, orientation_path,
                 calibration_path, device='cuda:0', max_reprojection_px=2.,
                 orientation_path_secondary=None, orientation_blend=.5,
                 pair_orientation=False):
        self.device = torch.device(device)
        self.locator = Locator(None).to(self.device).eval()
        self.locator.load_state_dict(torch.load(locator_path, map_location='cpu', weights_only=False)['model'])
        broad_checkpoint=torch.load(broad_path, map_location='cpu', weights_only=False)
        self.broad_condition_port=bool(broad_checkpoint.get('condition_target_port',False))
        self.broad_crop_size=int(broad_checkpoint.get('crop_size',480))
        self.broad_input_size=int(broad_checkpoint.get('network_input_size',224))
        self.broad = PretrainedKeypointRefiner(None, self.broad_condition_port,
                                               point_count=1).to(self.device).eval()
        self.broad.load_state_dict(broad_checkpoint['model'])
        self.fine = PretrainedKeypointRefiner(None, False, point_count=1).to(self.device).eval()
        self.fine.load_state_dict(torch.load(fine_path, map_location='cpu', weights_only=False)['model'])
        orientation = torch.load(orientation_path, map_location='cpu', weights_only=False)
        self.orientation = Model(orientation['model']['mean'], orientation['model']['std']).to(self.device).eval()
        self.orientation.load_state_dict(orientation['model'])
        self.orientation_secondary=None
        self.orientation_blend=float(orientation_blend)
        if orientation_path_secondary:
            other=torch.load(orientation_path_secondary,map_location='cpu',weights_only=False)
            self.orientation_secondary=Model(other['model']['mean'],other['model']['std']).to(self.device).eval()
            self.orientation_secondary.load_state_dict(other['model'])
        calibration = json.loads(Path(calibration_path).read_text())
        self.optical = {c: np.asarray(calibration['tcp_to_optical'][c], float) for c in CAMERAS}
        self.max_reprojection_px = float(max_reprojection_px)
        self.pair_orientation=bool(pair_orientation)

    def _crop_predictions(self, native, centers, model, *, crop_size, input_size,
                          port_index=None):
        images = []
        offsets = []
        for cam, center in zip(CAMERAS, centers):
            left = int(round(float(center[0]) - crop_size / 2))
            top = int(round(float(center[1]) - crop_size / 2))
            image = cv2.warpAffine(native[cam], np.array([[1, 0, -left], [0, 1, -top]], np.float32),
                                   (crop_size, crop_size), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT)
            if input_size != crop_size:
                image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_AREA)
            images.append(torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1))
            offsets.append([left, top])
        batch = torch.stack(images).to(self.device).float() / 255.
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.bfloat16,
                                                    enabled=self.device.type == 'cuda'):
            condition=None
            if model.condition_target_port:
                if port_index is None:raise ValueError('Target port required for conditioned refiner')
                condition=torch.zeros((len(batch),2),device=self.device)
                condition[:,port_index]=1
            predicted = model(batch,condition).float().cpu().numpy()
        return predicted.reshape(3, 2) * (crop_size - 1) + np.asarray(offsets)

    def predict(self, native, small, state, task_vector):
        card = int(np.argmax(task_vector[4:9]))
        port = int(np.argmax(task_vector[2:4]))
        identity = f'sfp_card_{card}_port_{port}'
        selected = NAMES.index(identity)
        images = torch.stack([torch.from_numpy(np.ascontiguousarray(small[c])).permute(2, 0, 1)
                              for c in CAMERAS]).to(self.device).float() / 255.
        meta = np.zeros((3, 5), np.float32)
        meta[:, 1] = 1
        for i in range(3):
            meta[i, 2+i] = 1
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.bfloat16,
                                                    enabled=self.device.type == 'cuda'):
            points, flags, features = self.locator(images, torch.as_tensor(meta, device=self.device),
                                                    return_features=True)
        points = points.float().cpu().numpy() * [1151, 1023]
        flags = flags.float().sigmoid().cpu().numpy()
        feature = features.float().cpu().numpy().reshape(-1)
        base_tcp = pose_matrix(state[:7])
        cameras = {c: base_tcp @ self.optical[c] for c in CAMERAS}
        candidates = []
        mask = []
        for j in range(len(NAMES)):
            pixels = {c: points[i, j] for i, c in enumerate(CAMERAS) if flags[i, j] >= .5}
            if len(pixels) >= 2:
                try:
                    position = triangulate(pixels, cameras)
                    valid = bool(np.isfinite(position).all())
                except np.linalg.LinAlgError:
                    valid = False
            else:
                valid = False
            candidates.extend(position.tolist() if valid else [0., 0., 0.])
            mask.append(float(valid))
        code = np.zeros(len(NAMES), np.float32)
        code[selected] = 1
        orientation_feature = np.r_[candidates, mask, code, state[:7], feature].astype(np.float32)
        with torch.inference_mode():
            rotation = self.orientation(torch.as_tensor(orientation_feature, device=self.device)[None]).float().cpu().numpy()[0]
            if self.orientation_secondary is not None:
                other=self.orientation_secondary(torch.as_tensor(orientation_feature,device=self.device)[None]).float().cpu().numpy()[0]
                first_rotation=Rotation.from_matrix(rotation)
                relative=first_rotation.inv()*Rotation.from_matrix(other)
                rotation=(first_rotation*Rotation.from_rotvec(relative.as_rotvec()*self.orientation_blend)).as_matrix()
        if any(flags[i, selected] < .5 for i in range(3)):
            return None, {'reason': 'selected_port_missing_projected_view', 'identity': identity,
                          'projected_inframe_not_visibility': True}
        coarse = points[:, selected]
        broad = self._crop_predictions(native, coarse, self.broad,
                                       crop_size=self.broad_crop_size,
                                       input_size=self.broad_input_size,port_index=port)
        fine = self._crop_predictions(native, broad, self.fine, crop_size=160, input_size=160)
        pixels = {c: fine[i] for i, c in enumerate(CAMERAS)}
        try:
            position = triangulate(pixels, cameras)
            residual = reprojection_residual_px(position, pixels, cameras)
        except np.linalg.LinAlgError:
            return None, {'reason': 'triangulation_failed', 'identity': identity}
        report = {'identity': identity, 'views': 3, 'reprojection_px': float(residual),
                  'coarse_pixels_px': coarse.tolist(),
                  'broad_pixels_px': broad.tolist(),
                  'fine_pixels_px': fine.tolist(),
                  'coarse_to_broad_px': np.linalg.norm(broad-coarse, axis=1).tolist(),
                  'broad_to_fine_px': np.linalg.norm(fine-broad, axis=1).tolist(),
                  'visibility_certified': False, 'runtime_true_geometry': False,
                  'reliability_gate': residual < self.max_reprojection_px}
        if not report['reliability_gate']:
            report['reason'] = 'rgb_views_disagree'
            return None, report
        if self.pair_orientation:
            # Fixed port spacing and mounting roll are asset priors shared by
            # all training scenes. The scene-specific yaw comes only from two
            # RGB-refined, triangulated openings; no simulator TF is read.
            sibling_guess=position+rotation@np.array([
                .0232 if port==1 else -.0232,0.,0.])
            sibling_matrix=np.eye(4);sibling_matrix[:3,3]=sibling_guess
            sibling_centers=[]
            for cam in CAMERAS:
                pixel=project(sibling_matrix,cameras[cam])
                if not pixel[3]:
                    report['reason']='sibling_not_in_frame'
                    return None,report
                sibling_centers.append(pixel[:2])
            # The selected and sibling mouths are ~90 px apart near contact.
            # A broad unconditioned refiner can snap to the selected mouth,
            # whereas the projected 160px crop isolates the sibling directly.
            sibling_fine=self._crop_predictions(native,sibling_centers,self.fine,
                crop_size=160,input_size=160)
            sibling_pixels={cam:sibling_fine[i] for i,cam in enumerate(CAMERAS)}
            candidates=[]
            for used in [CAMERAS,*combinations(CAMERAS,2)]:
                subset_pixels={cam:sibling_pixels[cam] for cam in used}
                subset_cameras={cam:cameras[cam] for cam in used}
                try:
                    candidate=triangulate(subset_pixels,subset_cameras)
                    residual=reprojection_residual_px(candidate,subset_pixels,subset_cameras)
                except np.linalg.LinAlgError:
                    continue
                displacement=position-candidate if port==0 else candidate-position
                spacing_candidate=float(np.linalg.norm(displacement))
                height_candidate=float(abs(displacement[2]))
                if (residual<self.max_reprojection_px and
                    .0202<=spacing_candidate<=.0262 and height_candidate<.003):
                    score=(abs(spacing_candidate-.0232)*1000+
                           height_candidate*1000+float(residual))
                    candidates.append((score,-len(used),candidate,residual,used))
            if not candidates:
                report.update(sibling_seed_pixels_px=np.asarray(sibling_centers).tolist(),
                              sibling_fine_pixels_px=sibling_fine.tolist(),
                              reason='sibling_pair_geometry_invalid')
                return None,report
            _,_,sibling_position,sibling_residual,used=min(candidates,key=lambda x:(x[1],x[0]))
            port0=position if port==0 else sibling_position
            port1=position if port==1 else sibling_position
            axis=port0-port1
            spacing=float(np.linalg.norm(axis))
            report.update(sibling_seed_pixels_px=np.asarray(sibling_centers).tolist(),
                sibling_fine_pixels_px=sibling_fine.tolist(),
                sibling_views_used=list(used),
                sibling_reprojection_px=float(sibling_residual),
                observed_port_spacing_mm=spacing*1000,
                sibling_height_difference_mm=float(abs(axis[2])*1000))
            yaw=float(np.arctan2(axis[1],axis[0]))
            rotation=Rotation.from_euler('xyz',[179.27564203264745,0.,np.rad2deg(yaw)],
                                         degrees=True).as_matrix()
            report['orientation_source']='observed_rgb_port_pair_and_fixed_mount_roll'
            report['pair_yaw_deg']=float(np.rad2deg(yaw))
        else:
            report['orientation_source']='learned_orientation_head'
        transform = np.eye(4)
        transform[:3, 3] = position
        transform[:3, :3] = rotation
        return transform, report
