"""Fixed five-shift rotation averaging, matching the trained crop augmentation."""
import time
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from native_connector_pose_observation import NativeConnectorPoseObservation

SHIFTS=((0,0),(-1,0),(1,0),(0,-1),(0,1))


class NativeConnectorOrientationTTA(NativeConnectorPoseObservation):
    @torch.inference_mode()
    def cropped_rotations(self, patches):
        patches=np.asarray(patches)
        if patches.ndim!=5 or patches.shape[1:]!=(3,192,192,3) or patches.dtype!=np.uint8:
            raise ValueError('Batched ordered three-camera native192 RGB crops required')
        padded=np.pad(patches,((0,0),(0,0),(1,1),(1,1),(0,0)),mode='edge')
        variants=np.stack([padded[:,:,1+dy:193+dy,1+dx:193+dx] for dx,dy in SHIFTS],axis=1)
        x=torch.from_numpy(variants.reshape(-1,3,192,192,3)).permute(0,1,4,2,3).to(self.device)
        _,rotation=self.model(x)
        rotations=rotation.cpu().numpy().reshape(len(patches),len(SHIFTS),3,3)
        means=np.stack([Rotation.from_matrix(r).mean().as_matrix() for r in rotations])
        spans=np.array([max(float(np.degrees(Rotation.from_matrix(a.T@b).magnitude())) for a in r for b in r) for r in rotations])
        return means,rotations[:,0],spans

    @torch.inference_mode()
    def predict(self, images, *, already_training_jpeg=False, visual_tip_position_tcp_m=None):
        if visual_tip_position_tcp_m is None:raise ValueError('Explicit image-derived position is required; no mean-position fallback')
        started=time.perf_counter();patches=[]
        for ci,camera in enumerate(['center','left','right']):
            rgb=images[camera]
            if rgb.shape!=(1024,1152,3) or rgb.dtype!=np.uint8:raise ValueError('Original RGB required')
            if not already_training_jpeg:
                ok,encoded=cv2.imencode('.jpg',rgb[...,::-1],[cv2.IMWRITE_JPEG_QUALITY,86])
                if not ok:raise ValueError('Training codec failed')
                rgb=cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
            x,y=self.origins[ci];patches.append(cv2.warpAffine(rgb,np.array([[1,0,-x],[0,1,-y]],np.float32),(192,192)))
        mean,_,span=self.cropped_rotations(np.stack(patches)[None]);position=np.asarray(visual_tip_position_tcp_m,dtype=float)
        if position.shape!=(3,) or not np.isfinite(position).all():raise ValueError('Invalid external RGB position')
        pose=np.eye(4);pose[:3,:3]=mean[0];pose[:3,3]=position
        return dict(predicted_connector_in_tcp_matrix=pose.tolist(),predicted_tip_position_tcp_m=position.tolist(),
            orientation_for_diagnostic_only=mean[0].tolist(),orientation_approved=False,runtime_true_geometry=False,
            orientation_postprocess='five_shift_so3_mean_v1',augmentation_rotation_span_deg=float(span[0]),
            translation_reestimated=False,latency_ms=(time.perf_counter()-started)*1000)
