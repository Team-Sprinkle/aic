"""Shared native-pixel opening refinement; no simulator geometry interface.

The caller supplies image-derived candidate centres and the public request.
Returns 2D pixels only. Calibrated multi-view triangulation and an independently
validated orientation estimate are still required for a full port pose.
"""
import cv2
import numpy as np
import torch

from train_joint_native_opening_refiner import JointRefiner, code, CAMS


class NativeRequestedOpeningRefiner:
    def __init__(self, checkpoint, device='cuda'):
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        schema=saved.get('schema');self.pose_keypoints=schema=='native_requested_pose_keypoints/v1'
        geometry=saved.get('geometry_input') if self.pose_keypoints else saved.get('runtime_geometry_input')
        if schema not in ('joint_native_opening_refiner/v1','native_requested_pose_keypoints/v1') or geometry is not False:
            raise ValueError('Expected observation-only shared native opening refiner checkpoint')
        self.reference_points=saved.get('reference_points')
        self.device=torch.device(device)
        if self.pose_keypoints:
            from train_native_requested_pose_keypoints import NativePoseKeypoints
            self.model=NativePoseKeypoints().to(device).eval()
        else:self.model=JointRefiner(None).to(device).eval()
        self.model.load_state_dict(saved['model'],strict=True)

    @torch.inference_mode()
    def refine(self, rgb_images, candidate_pixels, *, family, port, card=None):
        if family not in ('sc','sfp') or port not in (0,1):raise ValueError('Invalid public target request')
        if family=='sfp' and (card is None or not 0<=card<5):raise ValueError('SFP requires the public selected card')
        identity=f'sc_port_{port}' if family=='sc' else f'sfp_card_{card}_port_{port}'
        cameras=[c for c in CAMS if c in candidate_pixels];patches=[];origins=[];tasks=[]
        if not cameras:return dict(pixels={},shifts_px={})
        for camera in cameras:
            im=rgb_images[camera]
            if im.shape!=(1024,1152,3) or im.dtype!=np.uint8:
                raise ValueError('Refinement requires original 1152x1024 uint8 RGB, before resizing')
            center=np.asarray(candidate_pixels[camera],float)
            if center.shape!=(2,) or not np.isfinite(center).all():raise ValueError('Invalid RGB candidate')
            origin=np.round(center-80).astype(int);origins.append(origin)
            patches.append(cv2.warpAffine(im,np.array([[1,0,-origin[0]],[0,1,-origin[1]]],np.float32),(160,160)))
            tasks.append(code(dict(family=family),identity,camera))
        x=torch.from_numpy(np.stack(patches)).permute(0,3,1,2).to(self.device).float()/255
        task=torch.from_numpy(np.stack(tasks)).to(self.device)
        local=self.model(x,task).float().cpu().numpy()*159
        if self.pose_keypoints:
            keypoints=local+np.asarray(origins)[:,None];predicted=keypoints[:,0]
        else:keypoints=None;predicted=local+np.asarray(origins)
        return dict(pixels={c:predicted[i].tolist() for i,c in enumerate(cameras)},
            shifts_px={c:float(np.linalg.norm(predicted[i]-candidate_pixels[c])) for i,c in enumerate(cameras)},
            keypoints=({c:keypoints[i].tolist() for i,c in enumerate(cameras)} if keypoints is not None else None),
            runtime_geometry_input=False,prediction_kind='five reference landmarks of ONE requested port' if self.pose_keypoints else 'opening pixels; not full orientation')
