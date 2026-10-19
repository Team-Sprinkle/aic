"""Inference definitions extracted from train_sc_requested_history_locator.py; no training entry point."""
import torch
from torch import nn

class RequestedHistoryLocator(nn.Module):

    def __init__(self, use_history=True, num_requests=2):
        super().__init__()
        self.use_history = use_history
        self.visual = nn.Linear(256, 96)
        self.motion = nn.Linear(9, 96)
        self.site = nn.Parameter(torch.randn(3, 72, 96) * 0.01)
        self.history_site = nn.Parameter(torch.randn(3, 6, 96) * 0.01)
        self.age = nn.Parameter(torch.randn(10, 96) * 0.01)
        self.request = nn.Embedding(num_requests, 96)
        self.camera = nn.Parameter(torch.randn(3, 96) * 0.01)
        layer = nn.TransformerEncoderLayer(96, 4, 192, dropout=0.1, batch_first=True, norm_first=True)
        self.memory = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.attend = nn.MultiheadAttention(96, 4, dropout=0.1, batch_first=True)
        self.heat = nn.Sequential(nn.Linear(192, 96), nn.GELU(), nn.Linear(96, 1))
        self.flag = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, 1))

    def forward(self, current, history, motion, mask, request):
        b = len(current)
        now = self.visual(current.flatten(-2).transpose(-1, -2).float()) + self.site[None]
        memory = now.flatten(1, 2)
        padding = torch.zeros(b, 216, dtype=torch.bool, device=current.device)
        if self.use_history:
            past = self.visual(history.float()) + self.history_site[None, None] + self.age[None, :, None, None] + self.motion(motion)[:, :, None, None]
            memory = torch.cat((past.flatten(1, 3), memory), 1)
            padding = torch.cat(((~mask)[:, :, None].expand(-1, -1, 18).flatten(1), padding), 1)
        memory = self.memory(memory, src_key_padding_mask=padding)
        query = self.request(request)[:, None] + self.camera[None]
        answer, _ = self.attend(query, memory, memory, key_padding_mask=padding, need_weights=False)
        query = query + answer
        logits = self.heat(torch.cat((now, query[:, :, None].expand(-1, -1, 72, -1)), -1)).squeeze(-1)
        y, x = torch.meshgrid(torch.linspace(0, 1, 8, device=current.device), torch.linspace(0, 1, 9, device=current.device), indexing='ij')
        grid = torch.stack((x.flatten(), y.flatten()), -1)
        point = torch.einsum('bck,kd->bcd', logits.softmax(-1), grid)
        return (point, self.flag(query).squeeze(-1), logits)
