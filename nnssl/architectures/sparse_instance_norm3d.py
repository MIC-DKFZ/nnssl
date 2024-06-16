
from torch import nn
import torch

class SparseInstanceNorm3d(nn.InstanceNorm3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = 0.1
        self.eps = 1e-5

    def forward(self, x, mask):
        # Compute the mean and variance of the non-zero elements
        n_mask = torch.sum(mask)
        non_zero_mean = torch.sum(x * mask) / n_mask
        var = torch.sum((x - non_zero_mean) ** 2 * mask) / (n_mask - 1)
        x = ((x - non_zero_mean) / torch.sqrt(var + self.eps)) * mask

        return x