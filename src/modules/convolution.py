import torch
from torch import nn
from torch.nn import init
import math


class SimpleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.reset_parameters()

    def forward(self, x):
        assert x.ndim == 4
        convolved = (self.weight[None, :, :, None, None] * x[:, :, None, :, :]).sum(dim=1)

        return convolved

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
