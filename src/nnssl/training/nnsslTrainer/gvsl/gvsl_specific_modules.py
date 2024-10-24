import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np


class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()

    def forward(self, src, flow, mode="bilinear"):
        shape = flow.shape[2:]

        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        grid = grid.cuda()

        new_locs = grid + flow

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, mode=mode)


class AffineTransformer(nn.Module):
    """
    3-D Affine Transformer
    """

    def __init__(self):
        super().__init__()

    def forward(self, src, mat, mode="bilinear"):
        norm = torch.tensor(
            [[1, 1, 1, src.shape[2]], [1, 1, 1, src.shape[3]], [1, 1, 1, src.shape[4]]], dtype=torch.float
        ).cuda()
        norm = norm[np.newaxis, :, :]
        mat_new = mat / norm
        grid = nnf.affine_grid(mat_new, [src.shape[0], 3, src.shape[2], src.shape[3], src.shape[4]])
        return nnf.grid_sample(src, grid, mode=mode)


class Re_SpatialTransformer(nn.Module):
    def __init__(self):
        super(Re_SpatialTransformer, self).__init__()
        self.stn = SpatialTransformer()

    def forward(self, src, flow, mode="bilinear"):
        flow = -1 * self.stn(flow, flow, mode="bilinear")

        return self.stn(src, flow, mode)
