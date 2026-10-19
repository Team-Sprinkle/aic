"""Inference definitions extracted from train_two_family_port_locator.py; no training entry point."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

class Locator(nn.Module):

    def __init__(self, weights):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        if weights is not None:
            self.backbone.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        for name, param in self.backbone.named_parameters():
            param.requires_grad_(name.startswith('layer3.'))
        self.decoder = nn.Sequential(nn.Conv2d(64 + 256 + 5, 128, 3, padding=1), nn.GELU(), nn.Conv2d(128, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 12, 1))
        self.inframe = nn.Sequential(nn.Linear(256 + 5, 128), nn.GELU(), nn.Linear(128, 12))
        self.register_buffer('mean', torch.tensor((0.485, 0.456, 0.406))[None, :, None, None])
        self.register_buffer('std', torch.tensor((0.229, 0.224, 0.225))[None, :, None, None])

    def train(self, mode=True):
        super().train(mode)
        self.backbone.bn1.eval()
        self.backbone.layer1.eval()
        self.backbone.layer2.eval()
        return self

    def forward(self, rgb, meta, return_features=False, return_heatmap=False, return_spatial=False):
        x = (rgb - self.mean) / self.std
        x = self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x))))
        early = self.backbone.layer1(x)
        deep = self.backbone.layer3(self.backbone.layer2(early))
        joined = torch.cat((early, F.interpolate(deep, size=early.shape[-2:], mode='bilinear', align_corners=False), meta[:, :, None, None].expand(-1, -1, *early.shape[-2:])), dim=1)
        heat = self.decoder(joined)
        b, k, h, w = heat.shape
        probs = F.softmax(heat.reshape(b, k, -1).float() * 4, dim=-1)
        y, x = torch.meshgrid(torch.linspace(0, 1, h, device=rgb.device), torch.linspace(0, 1, w, device=rgb.device), indexing='ij')
        coords = torch.stack(((probs * x.flatten()).sum(-1), (probs * y.flatten()).sum(-1)), -1)
        flag = self.inframe(torch.cat((deep.mean((-1, -2)), meta), -1))
        if return_features:
            feature = torch.cat((deep.mean((-1, -2)), self.backbone.layer4(deep).mean((-1, -2))), dim=-1)
            if return_spatial:
                return (coords, flag, feature, deep)
            return (coords, flag, feature)
        if return_heatmap:
            return (coords, flag, heat)
        return (coords, flag)
