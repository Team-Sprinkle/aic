"""Longer complete port-frame action chunks; legacy four-query model intact."""
import torch
from torch import nn
from train_shared_port_tcp_bc import SharedPortTCPActor


class ChunkedPortActor(SharedPortTCPActor):
    def __init__(self, stats, *, action_horizon=4, **kwargs):
        if action_horizon not in (4, 8, 12):
            raise ValueError('Declared50ms horizons are4,8,12')
        if kwargs.get('decoder_kind', 'transformer') != 'transformer':
            raise ValueError('This matched horizon experiment uses the transformer')
        super().__init__(stats, **kwargs)
        self.action_horizon = action_horizon
        if action_horizon != 4:
            self.waypoint_tokens = nn.Parameter(torch.randn(1, action_horizon, 256) * .01)
