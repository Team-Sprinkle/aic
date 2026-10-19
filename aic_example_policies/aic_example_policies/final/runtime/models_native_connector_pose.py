"""Inference definitions extracted from train_native_connector_pose.py; no training entry point."""
import torch
from torch import nn
from torch.nn import functional as F
from .models_sc_native_keypoints import PretrainedKeypointRefiner

def rotation6(x):
    first = F.normalize(x[..., :3], dim=-1)
    second = F.normalize(x[..., 3:] - (first * x[..., 3:]).sum(-1, keepdim=True) * first, dim=-1)
    return torch.stack((first, second, torch.linalg.cross(first, second)), -1)

class ConnectorPose(nn.Module):

    def __init__(self, encoder_state, mean, scale):
        super().__init__()
        self.landmark = PretrainedKeypointRefiner(None, False, point_count=5, condition_camera=True)
        self.landmark.backbone.layer4 = nn.Identity()
        self.landmark.backbone.avgpool = nn.Identity()
        self.landmark.backbone.fc = nn.Identity()
        self.landmark.load_state_dict(encoder_state, strict=True)
        for parameter in self.landmark.decoder.parameters():
            parameter.requires_grad_(False)
        self.project = nn.Conv2d(256, 32, 1)
        self.head = nn.Sequential(nn.LayerNorm(3 * 32 * 6 * 6), nn.Linear(3 * 32 * 6 * 6, 256), nn.GELU(), nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 9))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer('pose_mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('pose_scale', torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, x):
        batch = len(x)
        x = x.reshape(-1, 3, 192, 192).float() / 255.0
        x = (x - self.landmark.mean) / self.landmark.std
        b = self.landmark.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        x = b.layer3(b.layer2(b.layer1(x)))
        x = F.adaptive_avg_pool2d(self.project(x), (6, 6)).reshape(batch, -1)
        pose = self.head(x) * self.pose_scale + self.pose_mean
        return (pose[:, :3], rotation6(pose[:, 3:]))
