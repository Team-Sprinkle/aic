"""Frozen SC plug position from original-image crops and fixed camera calibration.

Only RGB, known camera identity and training-derived fixed crop centres enter
inference. Simulator transforms and scene files are deliberately absent.
"""
import time
import cv2,numpy as np,torch
from torch import nn
from train_sc_native_keypoints import PretrainedKeypointRefiner
from evaluate_sc_native_triangulation import triangulate


class SCNativeGraspObservation:
    def __init__(self,path,device='cuda'):
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved['schema']!='sc_native_grasp_translation/v1' or saved['label_time_contract']!='observation_controller_stamp':
            raise ValueError('Expected corrected observation-time translation-only export')
        self.device=torch.device(device);self.cameras=('center','left','right')
        self.centres=saved['crop_centres'];self.optics={c:np.asarray(saved['tcp_to_optical'][c]) for c in self.cameras};self.cad=np.asarray(saved['CAD_reference_points'])
        self.model=PretrainedKeypointRefiner(None,False,point_count=5,condition_camera=True)
        self.model.backbone.layer4=nn.Identity();self.model.backbone.avgpool=nn.Identity();self.model.backbone.fc=nn.Identity()
        self.model.load_state_dict(saved['model'],strict=True);self.model.to(self.device).eval();self.latency_ms=[]

    @torch.inference_mode()
    def predict(self,images,*,already_training_jpeg=False):
        start=time.perf_counter();patches=[];offsets=[]
        for c in self.cameras:
            image=images[c]
            if image.shape!=(1024,1152,3):raise ValueError('Original camera pixels are required')
            if not already_training_jpeg:
                ok,encoded=cv2.imencode('.jpg',image[...,::-1],[cv2.IMWRITE_JPEG_QUALITY,86])
                if not ok:raise RuntimeError('Native training codec failed')
                image=cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
            left,top=np.rint(np.asarray(self.centres[c])-96).astype(int)
            patches.append(cv2.warpAffine(image,np.array([[1,0,-left],[0,1,-top]],np.float32),(192,192)))
            offsets.append([left,top])
        x=torch.from_numpy(np.stack(patches)).permute(0,3,1,2).to(self.device).float()/255
        pixels=self.model(x,camera_id=torch.eye(3,device=self.device)).cpu().numpy().reshape(3,5,2)*191+np.asarray(offsets)[:,None,:]
        cloud=np.stack([triangulate({c:pixels[i,k] for i,c in enumerate(self.cameras)},self.optics) for k in range(5)])
        a=self.cad-self.cad.mean(0);b=cloud-cloud.mean(0);u,_,vh=np.linalg.svd(a.T@b);rotation=vh.T@np.diag([1,1,np.linalg.det(vh.T@u.T)])@u.T
        position=cloud.mean(0)-rotation@self.cad.mean(0);duration=(time.perf_counter()-start)*1000;self.latency_ms.append(duration)
        return {'predicted_tip_position_tcp_m':position.tolist(),'predicted_pixels':pixels.tolist(),
                'orientation_for_diagnostic_only':rotation.tolist(),'orientation_approved':False,
                'translation_only':True,'runtime_true_geometry':False,'latency_ms':duration,
                'input':'three original192 crops at training-only fixed centres + camera identity'}
