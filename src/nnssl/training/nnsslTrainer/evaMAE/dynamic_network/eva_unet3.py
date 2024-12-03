from typing import Tuple

import torch
from nnssl.training.nnsslTrainer.evaMAE.dynamic_network.eva import Eva
from nnssl.training.nnsslTrainer.evaMAE.dynamic_network.vit_embed_decode import PatchEmbed, PatchDecode, LayerNormNd
from nnssl.training.nnsslTrainer.evaMAE.dynamic_network.weight_init import InitWeights_He
from einops import rearrange
from timm.layers import RotaryEmbeddingCat
from torch import nn


class EvaUNet3(nn.Module):
    def __init__(self,
                 input_channels: int,
                 embed_dim: int,
                 patch_embed_size: Tuple[int, ...],
                 output_channels: int,
                 eva_depth: int = 24,
                 eva_numheads: int = 16,
                 input_shape: Tuple[int, ...] = None,
                 decoder_norm=LayerNormNd,
                 decoder_act=nn.GELU,
                 num_register_tokens: int = 0,
                 use_rot_pos_emb: bool = True,
                 use_abs_pos_embed: bool = True,
                 mlp_ratio = 4 * 2 / 3,
                 drop_path_rate = 0,  # drops computations (multihead attention, mlp), Implementation of scaling might be useless here because this is not batch normed
                 drop_path_scale: bool = True,
                 patch_drop_rate: float = 0.,  # drops input patches, may be used for MAE style pretraining
                 proj_drop_rate: float = 0.,  # drops out things related to the projection. That is in the MLP and at the end of EVA attention
                 attn_drop_rate: float = 0.,  # drops attention, meaning connections between patches may bebroken up at random
                 rope_impl=RotaryEmbeddingCat,
                 rope_kwargs=None
                 ):
        """
        consists of a UNet encoder, a EVA ViT bottleneck and a UNet decoder
        """
        assert input_shape is not None
        assert len(input_shape) == 3, "Currently on ly 3d is supported"
        assert all([j % i == 0 for i, j in zip(patch_embed_size, input_shape)])

        super().__init__()

        self.down_projection = PatchEmbed(patch_embed_size, input_channels, embed_dim)
        self.up_projection = PatchDecode(patch_embed_size, embed_dim, output_channels,
                                         norm=decoder_norm,
                                         activation=decoder_act)

        # we need to compute the ref_feat_shape for eva
        self.eva = Eva(
            embed_dim=embed_dim,
            depth=eva_depth,
            num_heads=eva_numheads,
            ref_feat_shape=tuple([i // ds for i, ds in zip(input_shape, patch_embed_size)]),
            num_reg_tokens=num_register_tokens,
            use_rot_pos_emb=use_rot_pos_emb,
            use_abs_pos_emb=use_abs_pos_embed,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            drop_path_scale=drop_path_scale,
            patch_drop_rate=patch_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            rope_impl=rope_impl,
            rope_kwargs=rope_kwargs
        )

        if num_register_tokens > 0:
            self.register_tokens = (
                nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
            )
            nn.init.normal_(self.register_tokens, std=1e-6)
        else:
            self.register_tokens = None

        self.down_projection.apply(InitWeights_He(1e-2))
        self.up_projection.apply(InitWeights_He(1e-2))
        # eva has its own initialization

    def forward(self, x):
        x = self.down_projection(x)
        # last output of the encoder is the input to EVA
        B, C, W, H, D = x.shape
        x = rearrange(x, 'b c w h d -> b (h w d) c')
        if self.register_tokens is not None:
            x = torch.cat(
                (
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x,
                ),
                dim=1,
            )
        x = self.eva(x)
        if self.register_tokens is not None:
            x = x[:, self.register_tokens.shape[1]:]
        x = rearrange(x, 'b (h w d) c -> b c w h d', h=H, w=W, d=D)

        dec_out = self.up_projection(x)
        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")


if __name__ == '__main__':
    a = nn.Conv3d(3, 4, 4, 4)
    b = nn.ConvTranspose3d(4, 2, 4, 4)
    c = torch.rand((2, 3, 16, 15, 16))
