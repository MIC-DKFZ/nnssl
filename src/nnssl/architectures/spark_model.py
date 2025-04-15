# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pprint import pformat
from typing import List

import sys
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

from nnssl.architectures.spark_utils import (
    SparseInstanceNorm3d,
    _get_active_ex_or_ii,
)


class SparK3D(nn.Module):

    def __init__(
        self,
        architecture: ResidualEncoderUNet,
        use_mask_token: bool = True,
    ):
        super().__init__()
        self.architecture = architecture
        self.encoder = architecture.encoder
        self.decoder = architecture.decoder
        self.use_mask_token = use_mask_token
        if self.use_mask_token:
            # ------- We need all this additional stuff only if we use mask tokens ------- #
            features_per_stage = architecture.encoder.output_channels
            strides = architecture.encoder.strides
            absolute_reduction_factor = []
            reduction_factor = (1, 1, 1)
            for s in strides:
                reduction_factor = (
                    reduction_factor[0] * abs(s[0]),
                    reduction_factor[1] * abs(s[1]),
                    reduction_factor[2] * abs(s[2]),
                )
                absolute_reduction_factor.append(reduction_factor)

            downsample_raito = absolute_reduction_factor
            self.downsample_raito = downsample_raito

            self.hierarchy = len(features_per_stage)

            self.densify_norm_str = "in"
            self.densify_norms = nn.ModuleList()
            self.densify_projs = nn.ModuleList()
            self.mask_tokens = nn.ParameterList()
            # build the `densify` layers
            for i, feats_per_stage in enumerate(features_per_stage):
                # from the smallest feat map to the largest; i=0: the last feat map; i=1: the second last feat map ...
                # create mask token
                p = nn.Parameter(torch.zeros(1, feats_per_stage, 1, 1, 1))
                trunc_normal_(p, mean=0, std=0.02, a=-0.02, b=0.02)
                self.mask_tokens.append(p)

                # create densify norm
                if self.densify_norm_str == "in":
                    densify_norm = SparseInstanceNorm3d(feats_per_stage)
                elif self.densify_norm_str == "bn":
                    raise NotImplementedError("Not implemented other BN in densification yet.")
                else:
                    densify_norm = nn.Identity()
                self.densify_norms.append(densify_norm)

                # create densify proj
                if i == 0:  # Always e_width = d_width in nnU-Net and e_width == d_width:
                    densify_proj = (
                        nn.Identity()
                    )  # todo: NOTE THAT CONVNEXT-S WOULD USE THIS, because it has a width of 768 that equals to the decoder's width 768
                    print(f"[SparK.__init__, densify {i+1}/{self.hierarchy}]: use nn.Identity() as densify_proj")
                else:
                    # Need to
                    kernel_size = 1 if i <= 0 else 3
                    densify_proj = nn.Conv3d(
                        feats_per_stage,
                        feats_per_stage,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                        bias=True,
                    )  #         ^^^^^^^ used to be d_width (which we don't differentiate by!
                    print(
                        f"[SparK.__init__, densify {i+1}/{self.hierarchy}]: densify_proj(ksz={kernel_size}, #para={sum(x.numel() for x in densify_proj.parameters()) / 1e6:.2f}M)"
                    )
                self.densify_projs.append(densify_proj)

            print(f"[SparK.__init__] dims of mask_tokens={tuple(p.numel() for p in self.mask_tokens)}")

    def forward(self, x: torch.Tensor):
        # step1. Mask

        # step2. Encode: get hierarchical encoded sparse features (a list containing 4 feature maps at 4 scales)
        fea_bcffs: List[torch.Tensor] = self.encoder(x)
        # fea_bcffs.reverse()  # after reversion: from the smallest feature map to the largest

        # step3. Densify: get hierarchical dense features for decoding
        # ---------------- Without Mask Token no densification is done --------------- #
        if self.use_mask_token:
            to_dec = []
            for i, bcfff in enumerate(fea_bcffs):  # from the smallest feature map to the largest
                if bcfff is not None:
                    bcfff = self.densify_norms[i](bcfff)
                    mask_tokens = self.mask_tokens[i].expand_as(bcfff)
                    cur_shape = [bcfff.shape[0]] + list(bcfff.shape[2:])  # B H W D
                    bcfff = torch.where(
                        _get_active_ex_or_ii(*cur_shape, device=bcfff.device, dtype=bcfff.dtype).expand_as(bcfff)
                        == 1,
                        bcfff,
                        mask_tokens,
                    )  # fill in empty (non-active) positions with [mask] tokens
                    bcfff: torch.Tensor = self.densify_projs[i](bcfff)
                to_dec.append(bcfff)
        else:
            to_dec = fea_bcffs

        # step4. Decode and reconstruct
        rec_bchw = self.decoder(to_dec)
        return rec_bchw


# The current MaskToken makes training so much slower and messes up Gradient Calculations.
#   Instead I propose to use the Mask Token just as a filler (without any Conv2d or Norm)
#   The decoder has to figure out how to handle the Token instead.
class EfficientSpark3D(nn.Module):

    def __init__(
        self,
        architecture: ResidualEncoderUNet,
        **kwargs,
    ):
        super().__init__()
        self.architecture = architecture
        self.encoder = architecture.encoder
        self.decoder = architecture.decoder

        # ------- We need all this additional stuff only if we use mask tokens ------- #
        features_per_stage = architecture.encoder.output_channels
        strides = architecture.encoder.strides
        absolute_reduction_factor = []
        reduction_factor = (1, 1, 1)
        for s in strides:
            reduction_factor = (
                reduction_factor[0] * abs(s[0]),
                reduction_factor[1] * abs(s[1]),
                reduction_factor[2] * abs(s[2]),
            )
            absolute_reduction_factor.append(reduction_factor)

        downsample_raito = absolute_reduction_factor
        self.downsample_raito = downsample_raito

        self.hierarchy = len(features_per_stage)

        self.mask_tokens = nn.ParameterList()
        # build the `densify` layers
        for i, feats_per_stage in enumerate(features_per_stage):
            # from the smallest feat map to the largest; i=0: the last feat map; i=1: the second last feat map ...
            # create mask token
            p = nn.Parameter(torch.zeros(1, feats_per_stage, 1, 1, 1))
            trunc_normal_(p, mean=0, std=0.02, a=-0.02, b=0.02)
            self.mask_tokens.append(p)

    def forward(self, x: torch.Tensor):
        # step1. Mask

        # step2. Encode: get hierarchical encoded sparse features (a list containing 4 feature maps at 4 scales)
        fea_bcffs: List[torch.Tensor] = self.encoder(x)
        # fea_bcffs.reverse()  # after reversion: from the smallest feature map to the largest

        # step3. Densify: get hierarchical dense features for decoding
        # ---------------- Without Mask Token no densification is done --------------- #
        to_dec = []
        for i, bcfff in enumerate(fea_bcffs):  # from the smallest feature map to the largest
            if bcfff is not None:
                # bcfff = self.densify_norms[i](bcfff)  # Why even norm?
                mask_tokens = self.mask_tokens[i]
                cur_shape = [bcfff.shape[0]] + list(bcfff.shape[2:])  # B H W D
                cur_mask = _get_active_ex_or_ii(*cur_shape, device=bcfff.device, dtype=bcfff.dtype)
                densification_mask = mask_tokens * (1 - cur_mask)
                bcfff = bcfff + densification_mask  # fill in empty (non-active) positions with [mask] tokens
                # bcfff: torch.Tensor = self.densify_projs[i](bcfff)  -- No projection because Why do this at all?
            to_dec.append(bcfff)

        # step4. Decode and reconstruct
        rec_bchw = self.decoder(to_dec)
        return rec_bchw
