"""SFP plug frame from three fixed original-image crops; no scene/TF input.

Translation and orientation require separate held-out approvals. Consumers
must not use the diagnostic orientation when its gate has not passed.
"""
import time
import cv2
import numpy as np
import torch
from torch import nn
from train_sc_native_keypoints import PretrainedKeypointRefiner
from train_sfp_native_grasp_keypoints import poses_from_pixels


class SFPNativeGraspObservation:
    def __init__(self, checkpoint, approval, device='cuda'):
        import json,hashlib
        from pathlib import Path
        summary=json.loads(Path(approval).read_text())
        if not summary.get('translation_approved'):raise ValueError('Held-out native grasp translation gate must pass')
        if summary.get('checkpoint_sha256')!=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest():
            raise ValueError('Grasp approval does not belong to this exact checkpoint')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False);self.plan=saved['plan']
        if self.plan.get('geometry_input') is not False or self.plan.get('family')!='sfp':raise ValueError('Expected RGB-only SFP native grasp model')
        self.device=torch.device(device);self.cameras=('center','left','right')
        self.orientation_approved=bool(summary.get('orientation_approved'))
        self.minimum_tcp_gap_mm=summary.get('minimum_estimated_TCP_axial_gap_mm')
        self.model=PretrainedKeypointRefiner(None,False,point_count=5,condition_camera=True)
        self.model.backbone.layer4=nn.Identity();self.model.backbone.avgpool=nn.Identity();self.model.backbone.fc=nn.Identity()
        self.model.load_state_dict(saved['model']);self.model.to(device).eval();self.latency_ms=[]

    @torch.inference_mode()
    def predict(self,images,*,already_training_jpeg=False,estimated_TCP_axial_gap_mm=None):
        if self.minimum_tcp_gap_mm is not None and (estimated_TCP_axial_gap_mm is None or
                not np.isfinite(estimated_TCP_axial_gap_mm) or estimated_TCP_axial_gap_mm<self.minimum_tcp_gap_mm):
            raise ValueError('This grasp model is approved only for free-approach acquisition before a static lock')
        started=time.perf_counter();patches=[];origins=[]
        for c in self.cameras:
            rgb=images[c]
            if rgb.shape!=(1024,1152,3) or rgb.dtype!=np.uint8:raise ValueError('Original uint8 RGB required')
            if not already_training_jpeg:
                ok,encoded=cv2.imencode('.jpg',rgb[...,::-1],[cv2.IMWRITE_JPEG_QUALITY,86])
                if not ok:raise RuntimeError('Native training codec normalization failed')
                rgb=cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
            # Validation uses central192 from fixed208 crop. Keep the exact
            # integer208 origin rather than independently rounding192.
            origin=np.rint(np.asarray(self.plan['crop_centres_train_only'][c])-104).astype(int)+8
            patches.append(cv2.warpAffine(rgb,np.array([[1,0,-origin[0]],[0,1,-origin[1]]],np.float32),(192,192)));origins.append(origin)
        image=torch.from_numpy(np.stack(patches)).permute(0,3,1,2).to(self.device).float()/255
        prediction=self.model(image,camera_id=torch.eye(3,device=self.device)).cpu().numpy().reshape(3,5,2)*191+np.asarray(origins)[:,None,:]
        pose=poses_from_pixels(prediction[None],self.plan['tcp_to_optical'],self.plan['reference_points'])[0]
        duration=(time.perf_counter()-started)*1000;self.latency_ms.append(duration)
        return dict(predicted_tip_position_tcp_m=pose[:3,3].tolist(),
                    orientation_for_diagnostic_only=pose[:3,:3].tolist(),
                    orientation_approved=self.orientation_approved,predicted_pixels=prediction.tolist(),
                    translation_only=not self.orientation_approved,latency_ms=duration,
                    runtime_true_geometry=False,input='three fixed original192 RGB crops + camera identity')
