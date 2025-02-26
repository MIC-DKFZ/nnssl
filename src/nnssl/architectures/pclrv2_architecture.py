import math
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.nn import Conv3d, BatchNorm3d, ReLU, Sequential, BatchNorm1d, Linear, ModuleList, AdaptiveAvgPool3d
import torch.nn.functional as F
from nnssl.architectures.nsUNet import ResidualEncoderUNet_noskip


class PCLRv2Architecture(nn.Module):
    def __init__(self, network: ResidualEncoderUNet_noskip):
        super().__init__()
        self.network = network
        self.encoder = self.network.encoder
        self.gap = AdaptiveAvgPool3d(output_size=(1, 1, 1))

        self.features_per_mid_stage = self.encoder.output_channels[::-1][1:-1]   # [32, 64, 128, 256, 320, 320] -> [320, 256, 128, 64]
        self.n_mid_stages = len(self.features_per_mid_stage)

        pixel_restore_heads = []
        batch_norms = []
        predictor_heads = []

        for features in self.features_per_mid_stage:
            pixel_restore_heads.append(
                Sequential(
                    Conv3d(features, 1, kernel_size=3, padding=1),
                    BatchNorm3d(num_features=1, momentum=0.1, affine=True),
                    ReLU(inplace=True)
                )
            )
            batch_norms.append(
                BatchNorm1d(features)
            )
            predictor_heads.append(
                Sequential(
                    Linear(features, 2 * features),
                    BatchNorm1d(2 * features),
                    ReLU(inplace=True),
                    Linear(2 * features, features)
                )
            )

        self.pixel_restore_heads = ModuleList(pixel_restore_heads)
        self.batch_norms = ModuleList(batch_norms)
        self.predictor_heads = ModuleList(predictor_heads)


    def forward(self, imgs: torch.Tensor, embeddings_only: bool = False):
        reconstructions, mid_outputs = self.network(imgs)   # deep supervision

        if not embeddings_only:
            mid_reconstructions, embeddings = [], []
            for i in range(len(mid_outputs)):
                mid_output = mid_outputs[i]
                mid_reconstruction = self.pixel_restore_heads[i](mid_output)
                mid_reconstruction = F.interpolate(mid_reconstruction, scale_factor=2**(self.n_mid_stages-i), mode="trilinear")
                pre_embedding = self.batch_norms[i](self.gap(mid_output).flatten(start_dim=1))
                post_embedding = self.predictor_heads[i](pre_embedding)

                mid_reconstructions.append(mid_reconstruction)
                embeddings.append((pre_embedding, post_embedding))
            return reconstructions, embeddings, mid_reconstructions

        embeddings = []
        for i in range(len(mid_outputs)):
            mid_output = mid_outputs[i]
            pre_embedding = self.batch_norms[i](self.gap(mid_output).flatten(start_dim=1))
            post_embedding = self.predictor_heads[i](pre_embedding)
            embeddings.append((pre_embedding, post_embedding))
        return embeddings





