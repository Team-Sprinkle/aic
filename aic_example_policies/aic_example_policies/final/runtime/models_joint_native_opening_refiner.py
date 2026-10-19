"""Inference definitions extracted from train_joint_native_opening_refiner.py; no training entry point."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .models_sc_native_keypoints import PretrainedKeypointRefiner
CAMS = ('center', 'left', 'right')

def code(row, identity, camera):
    result = np.zeros(13, np.float32)
    result[0 if row['family'] == 'sfp' else 1] = 1
    result[2 + int(identity.rsplit('_', 1)[1])] = 1
    if row['family'] == 'sfp':
        result[4 + int(identity.split('_')[2])] = 1
        result[9] = 1
    result[10 + CAMS.index(camera)] = 1
    return result

class JointRefiner(PretrainedKeypointRefiner):

    def __init__(self, imagenet=None):
        super().__init__(imagenet, False, point_count=1)
        self.decoder[0] = nn.Conv2d(64 + 256 + 13, 128, 3, padding=1)

    def forward(self, image, task):
        x = (image - self.mean) / self.std
        x = self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x))))
        early = self.backbone.layer1(x)
        deep = self.backbone.layer3(self.backbone.layer2(early))
        z = torch.cat((early, F.interpolate(deep, size=early.shape[-2:], mode='bilinear', align_corners=False), task[:, :, None, None].expand(-1, -1, *early.shape[-2:])), 1)
        heat = self.decoder(z).flatten(1).float()
        prob = (heat * 4).softmax(-1)
        side = early.shape[-1]
        y, x = torch.meshgrid(torch.linspace(0, 1, side, device=image.device), torch.linspace(0, 1, side, device=image.device), indexing='ij')
        return torch.stack(((prob * x.flatten()).sum(-1), (prob * y.flatten()).sum(-1)), -1)
