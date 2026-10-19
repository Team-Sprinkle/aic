"""Observation-only conservative empty-NIC-row score; not a collision planner."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18


class EmptyRowObservation:
    def __init__(self,checkpoint,approval,device='cuda'):
        certificate=json.loads(Path(approval).read_text())
        if (not certificate.get('expanded_offline_gate_passed') or
                not certificate.get('visual_inspection_completed') or
                certificate['checkpoint_sha256']!=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()):
            raise ValueError('Exact empty-row component requires held-out and image review')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        backbone=resnet18(weights=None)
        self.encoder=nn.Sequential(backbone.conv1,backbone.bn1,backbone.relu,backbone.maxpool,backbone.layer1,backbone.layer2,backbone.layer3,nn.AdaptiveAvgPool2d((2,2))).to(device).eval()
        self.encoder.load_state_dict(saved['encoder'])
        self.head=nn.Sequential(nn.Linear(3072,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1)).to(device).eval();self.head.load_state_dict(saved['head'])
        self.device=device;self.feature_mean=torch.as_tensor(saved['feature_mean'],device=device);self.feature_scale=torch.as_tensor(saved['feature_scale'],device=device)
        self.mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];self.std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
        self.threshold=float(saved['threshold'])

    @torch.inference_mode()
    def predict(self,images,*,already_training_jpeg=False):
        frames=[]
        for camera in ('center','left','right'):
            image=images[camera]
            if not already_training_jpeg:
                ok,encoded=cv2.imencode('.jpg',image[:,:,::-1],[cv2.IMWRITE_JPEG_QUALITY,86])
                if not ok:raise ValueError('JPEG86 conversion failed')
                image=cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
            frames.append(cv2.resize(image,(224,224)))
        x=torch.from_numpy(np.stack(frames)).permute(0,3,1,2).to(self.device).float()/255
        feature=self.encoder((x-self.mean)/self.std).reshape(1,-1)
        probability=float(torch.sigmoid(self.head((feature-self.feature_mean)/self.feature_scale)).item())
        return dict(clear=probability>=self.threshold,score=probability,threshold=self.threshold,
                    runtime_true_geometry=False,not_calibrated_collision_probability=True)
