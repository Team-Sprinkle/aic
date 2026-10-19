"""Orientation model retaining finer spatial features from all three views.

The extra path starts at zero, preserving the complete warm model's physical
prediction. This is a perception feature ablation, not an action correction.
"""
import torch
from torch import nn
from torch.nn import functional as F

from train_native_connector_pose import ConnectorPose, rotation6


class FineConnectorPose(ConnectorPose):
    def __init__(self, encoder_state, mean, scale):
        super().__init__(encoder_state, mean, scale)
        self.fine_project = nn.Conv2d(128, 8, 1)
        self.fine_head = nn.Sequential(nn.LayerNorm(3*8*12*12),
                                       nn.Linear(3*8*12*12, 128), nn.GELU(),
                                       nn.Linear(128, 6))
        nn.init.zeros_(self.fine_head[-1].weight)
        nn.init.zeros_(self.fine_head[-1].bias)

    def forward(self, x):
        batch = len(x)
        x = x.reshape(-1, 3, 192, 192).float()/255.
        x = (x-self.landmark.mean)/self.landmark.std
        b = self.landmark.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        fine = b.layer2(b.layer1(x))
        coarse = b.layer3(fine)
        coarse = F.adaptive_avg_pool2d(self.project(coarse), (6, 6)).reshape(batch, -1)
        fine = F.adaptive_avg_pool2d(self.fine_project(fine), (12, 12)).reshape(batch, -1)
        pose = self.head(coarse)*self.pose_scale+self.pose_mean
        rotation_coordinates = pose[:, 3:]+self.fine_head(fine)*self.pose_scale[3:]
        return pose[:, :3], rotation6(rotation_coordinates)

    def load_coarse_warm_start(self, state):
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(not k.startswith(('fine_project.', 'fine_head.'))
                                         for k in result.missing_keys):
            raise ValueError('Warm start differs outside the new fine-feature branch')
        return result
