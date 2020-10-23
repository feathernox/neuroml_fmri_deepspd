import geoopt
import torch
import torch.nn.init
from torch import nn
import numpy as np


class AutocorrelationConstruction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        len_series = (~torch.isnan(input)).sum(dim=-1)[..., 0]
        input_mean = input[~torch.isnan(input)].mean(dim=-1)
        input_std = input[~torch.isnan(input)].std(dim=-1)
        input = (input - input_mean[..., None]) / input_std[..., None]
        input[torch.isnan(input)] = 0

        corr = torch.matmul(input, input.transpose(-1, -2)) / len_series[..., None, None]
        return corr


class BiMap(nn.Module):
    """
    Huang, Z., & Gool, L. (2017). A Riemannian Network for SPD Matrix Learning. ArXiv, abs/1608.04233.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        assert out_features <= in_features

        self.in_features = in_features
        self.out_features = out_features
        self.shape = (self.in_features, self.out_features)

        self.manifold = geoopt.Stiefel()
        self.weight = geoopt.ManifoldParameter(
            torch.empty(*self.shape), manifold=self.manifold
        )

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.weight.set_(self.manifold.random(self.shape))
        
    def forward(self, x):
        out = self.weight.T @ x @ self.weight
        return out


class ReEig(nn.Module):
    """
    Huang, Z., & Gool, L. (2017). A Riemannian Network for SPD Matrix Learning. ArXiv, abs/1608.04233.
    """
    def __init__(self, eps=0.1):
        super(ReEig, self).__init__()
        self.eps = eps
    
    def forward(self, x):
        L, U = torch.symeig(x, eigenvectors=True)
        L = torch.max(L, torch.ones_like(L) * self.eps)
        out = torch.matmul(U, torch.matmul(L.diag_embed(), U.transpose(-2, -1)))
        return out


class ExpEig(nn.Module):
    def __init__(self):
        super(ExpEig, self).__init__()

    def forward(self, x):
        L, U = torch.symeig(x, eigenvectors=True)
        L = torch.exp(L)
        out = torch.matmul(U, torch.matmul(L.diag_embed(), U.transpose(-2, -1)))
        return out


class LogEig(nn.Module):
    def __init__(self, eps=1e-5):
        super(LogEig, self).__init__()
        self.eps = eps
    
    def forward(self, x):
        L, U = torch.symeig(x, eigenvectors=True)
        L = torch.max(L, torch.ones_like(L) * self.eps)
        L = torch.log(L)
        out = torch.matmul(U, torch.matmul(L.diag_embed(), U.transpose(-2, -1)))
        return out


class FlattenSPD(nn.Module):
    def __init__(self, input_size, main_diag=True, norm=False, sqrt2=False):
        super(FlattenSPD, self).__init__()

        self.input_size = input_size
        row_idx, col_idx = np.triu_indices(input_size, k=0 if main_diag else 1)
        self.register_buffer('row_idx', torch.LongTensor(row_idx))
        self.register_buffer('col_idx', torch.LongTensor(col_idx))

        self.norm = norm
        self.sqrt2 = sqrt2

    def forward(self, x):
        output = x[:, self.row_idx, self.col_idx]

        if self.sqrt2:
            output[..., self.row_idx != self.col_idx] *= np.sqrt(2)

        if self.norm:
            output = torch.sign(output) * torch.sqrt(torch.abs(output))
            output = output / torch.norm(output, dim=-1, keepdim=True)
        return output

    def extra_repr(self):
        return f'input_size={self.input_size}, norm={self.norm}, sqrt2={self.sqrt2}'


class ConstantScaling(nn.Module):
    def __init__(self, min_constant=1e-3):
        super(ConstantScaling, self).__init__()
        self.constant = nn.Parameter(torch.empty((1,)))
        self.min_constant = min_constant
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.ones_(self.constant)

    def forward(self, x):
        x = x * torch.clamp(self.constant, min=self.min_constant)
        return x


class Drop(nn.Module):
    def __init__(self, in_features, drop_size):
        super().__init__()

        self.in_features = in_features
        self.drop_size = drop_size
        if isinstance(drop_size, int):
            self.drop_features = drop_size
        elif isinstance(drop_size, float):
            self.drop_features = round(drop_size * in_features)
        else:
            raise TypeError("drop_size must be int or float; got {0}".format(type(drop_size)))

    # TODO implement forward
    def forward(self, x):
        return x


class SMSOPooling(nn.Module):
    def __init__(self, in_features, out_features):
        super(SMSOPooling, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        assert x.shape[-2] == x.shape[-1] and x.shape[-1] == self.in_features

        z = (torch.matmul(self.weight, x) * self.weight).sum(dim=-1)
        z = torch.sqrt(z)

        return z
