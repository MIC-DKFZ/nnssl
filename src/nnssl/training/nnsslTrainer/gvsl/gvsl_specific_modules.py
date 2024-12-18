import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data, flow, mode="bilinear", padding_mode="zeros"):
        shape = flow.shape[2:]

        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack(grids)
        grid = grid.unsqueeze(0).float() #.cuda()

        deformable_map = grid + flow

        for i in range(len(shape)):
            deformable_map[:, i, ...] = 2 * (deformable_map[:, i, ...] / (shape[i] - 1) - 0.5)

        deformable_map = deformable_map.permute(0, 2, 3, 4, 1)
        deformable_map = deformable_map[..., [2, 1, 0]]

        return F.grid_sample(data, deformable_map, mode=mode, padding_mode=padding_mode, align_corners=False)


class AffineTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data, affine_mat, mode="bilinear"):
        norm = torch.tensor(
            [[1, 1, 1, data.shape[2]], [1, 1, 1, data.shape[3]], [1, 1, 1, data.shape[4]]], dtype=torch.float32
        ).unsqueeze(0) #.cuda()
        norm_affine_mat = affine_mat / norm
        grid = F.affine_grid(norm_affine_mat, [data.shape[0], 3, data.shape[2], data.shape[3], data.shape[4]], align_corners=False)
        return F.grid_sample(data, grid, mode=mode, align_corners=False)

