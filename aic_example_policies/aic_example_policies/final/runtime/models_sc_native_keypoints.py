"""Inference definitions extracted from train_sc_native_keypoints.py; no training entry point."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

class PretrainedKeypointRefiner(nn.Module):
    """ImageNet ResNet-18 spatial features with a compact 56x56 heatmap head."""

    def __init__(self, pretrained_path, condition_target_port=False, point_count=2, offscreen_opening_head=False, condition_camera=False):
        super().__init__()
        self.condition_target_port = condition_target_port
        self.point_count = point_count
        self.offscreen_opening_head = offscreen_opening_head
        self.condition_camera = condition_camera
        if offscreen_opening_head and (not condition_target_port or point_count < 2):
            raise ValueError('Offscreen opening head requires a target port and two points')
        self.backbone = models.resnet18(weights=None)
        if pretrained_path is not None:
            self.backbone.load_state_dict(torch.load(pretrained_path, map_location='cpu', weights_only=True))
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad_(name.startswith('layer3.'))
        self.decoder = nn.Sequential(nn.Conv2d(64 + 256 + (2 if condition_target_port else 0) + (3 if condition_camera else 0), 128, 3, padding=1), nn.GELU(), nn.Conv2d(128, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, point_count, 1))
        if offscreen_opening_head:
            self.opening_offset = nn.Sequential(nn.Linear(256 + 2, 128), nn.GELU(), nn.Linear(128, 2))
            nn.init.zeros_(self.opening_offset[-1].weight)
            nn.init.zeros_(self.opening_offset[-1].bias)
        self.register_buffer('mean', torch.tensor((0.485, 0.456, 0.406))[None, :, None, None])
        self.register_buffer('std', torch.tensor((0.229, 0.224, 0.225))[None, :, None, None])

    def train(self, mode=True):
        super().train(mode)
        self.backbone.bn1.eval()
        self.backbone.layer1.eval()
        self.backbone.layer2.eval()
        return self

    def forward(self, image, target_port=None, camera_id=None):
        x = (image - self.mean) / self.std
        x = self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x))))
        early = self.backbone.layer1(x)
        middle = self.backbone.layer2(early)
        deep = self.backbone.layer3(middle)
        features = [early, F.interpolate(deep, size=early.shape[-2:], mode='bilinear', align_corners=False)]
        if self.condition_target_port:
            if target_port is None or target_port.shape != (image.shape[0], 2):
                raise ValueError('Two-element target-port one-hot required')
            features.append(target_port[:, :, None, None].expand(-1, -1, *early.shape[-2:]))
        if self.condition_camera:
            if camera_id is None or camera_id.shape != (image.shape[0], 3):
                raise ValueError('Three-element observed camera identity required')
            features.append(camera_id[:, :, None, None].expand(-1, -1, *early.shape[-2:]))
        heatmap = self.decoder(torch.cat(features, dim=1))
        batch, points, height, width = heatmap.shape
        prob = torch.softmax(heatmap.reshape(batch, points, -1) * 4, dim=-1)
        y, x = torch.meshgrid(torch.linspace(0, 1, height, device=image.device), torch.linspace(0, 1, width, device=image.device), indexing='ij')
        x = x.reshape(-1)
        y = y.reshape(-1)
        coordinates = torch.stack(((prob * x).sum(-1), (prob * y).sum(-1)), dim=-1).reshape(batch, 2 * self.point_count)
        if self.offscreen_opening_head:
            offset = self.opening_offset(torch.cat((deep.mean((-1, -2)), target_port), dim=-1))
            coordinates = torch.cat((coordinates[:, :2], coordinates[:, 2:4] + offset, coordinates[:, 4:]), dim=-1)
        return coordinates
