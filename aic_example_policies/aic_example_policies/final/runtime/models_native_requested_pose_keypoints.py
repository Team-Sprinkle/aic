"""Inference definitions extracted from train_native_requested_pose_keypoints.py; no training entry point."""
import torch
from torch import nn
from torch.nn import functional as F
from .models_joint_native_opening_refiner import JointRefiner

class NativePoseKeypoints(JointRefiner):

    def __init__(self, imagenet=None):
        super().__init__(imagenet)
        self.decoder[-1] = nn.Conv2d(64, 5, 1)

    def forward(self, image, task):
        x = (image - self.mean) / self.std
        b = image.shape[0]
        x = self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x))))
        early = self.backbone.layer1(x)
        deep = self.backbone.layer3(self.backbone.layer2(early))
        z = torch.cat((early, F.interpolate(deep, size=early.shape[-2:], mode='bilinear', align_corners=False), task[:, :, None, None].expand(-1, -1, *early.shape[-2:])), 1)
        probability = (self.decoder(z).reshape(b, 5, -1).float() * 4).softmax(-1)
        side = early.shape[-1]
        y, x = torch.meshgrid(torch.linspace(0, 1, side, device=image.device), torch.linspace(0, 1, side, device=image.device), indexing='ij')
        return torch.stack(((probability * x.flatten()).sum(-1), (probability * y.flatten()).sum(-1)), -1)
