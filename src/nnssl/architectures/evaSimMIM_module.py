from typing import Tuple

import torch
from dynamic_network_architectures.building_blocks.eva import Eva
from dynamic_network_architectures.building_blocks.patch_encode_decode import PatchEmbed, PatchDecode, LayerNormNd
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from einops import rearrange
from timm.layers import RotaryEmbeddingCat
from torch import nn


class EvaSimMIM(nn.Module):
    def __init__(
        self,
        input_channels: int,
        embed_dim: int,
        patch_embed_size: Tuple[int, ...],
        output_channels: int,
        encoder_eva_depth: int = 24,
        encoder_eva_numheads: int = 16,
        input_shape: Tuple[int, ...] = None,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        num_register_tokens: int = 0,
        use_rot_pos_emb: bool = True,
        use_abs_pos_emb: bool = True,
        mlp_ratio=4 * 2 / 3,
        drop_path_rate=0,
        drop_path_scale: bool = True,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        rope_impl=RotaryEmbeddingCat,
        rope_kwargs=None,
        do_up_projection=True,
        init_values=None,
        scale_attn_inner=False,
    ):
        """
        Masked Autoencoder with EVA attention-based encoder and decoder.
        """
        assert input_shape is not None
        assert len(input_shape) == 3, "Currently only 3D is supported"
        assert all([j % i == 0 for i, j in zip(patch_embed_size, input_shape)])

        super().__init__()
        self.patch_embed_size = patch_embed_size
        self.embed_dim = embed_dim

        # Patch embedding for encoder
        self.down_projection = PatchEmbed(patch_embed_size, input_channels, embed_dim)

        # Encoder using EVA
        self.eva = Eva(
            embed_dim=embed_dim,
            depth=encoder_eva_depth,
            num_heads=encoder_eva_numheads,
            ref_feat_shape=tuple([i // ds for i, ds in zip(input_shape, patch_embed_size)]),
            num_reg_tokens=num_register_tokens,
            use_rot_pos_emb=use_rot_pos_emb,
            use_abs_pos_emb=use_abs_pos_emb,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            rope_impl=rope_impl,
            rope_kwargs=rope_kwargs,
            init_values=init_values,
            scale_attn_inner=scale_attn_inner,
        )

        # Patch embedding for decoder
        if do_up_projection:
            self.up_projection = PatchDecode(
                patch_embed_size, embed_dim, output_channels, norm=decoder_norm, activation=decoder_act
            )
        else:
            self.up_projection = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=1e-6)

        self.down_projection.apply(InitWeights_He(1e-2))
        self.down_projection.apply(InitWeights_He(1e-2))

    def apply_mask(self, x, mask):
        """
        Replace tokens in the input with the learnable mask token vector, locations are provided by the mask

        :param x: Input tensor of shape (B, num_tokens, embed_dim) representing token embeddings.
        :param mask: Boolean mask of shape (B, num_tokens, 1) where 0 indicates a masked token.
        :return: Masked input tensor of the same shape as `x`.
        """

        return x * mask + self.mask_token * (1 - mask)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # Encode patches
        x = self.down_projection(x)
        B, C, W, H, D = x.shape
        x = rearrange(x, "b c w h d -> b (w h d) c")  # (B, num_tokens, embed_dim)

        mask = mask.flatten(start_dim=2).transpose(1, 2)  # (B, num_tokens, 1)
        x = self.apply_mask(x, mask)

        # Encode using EVA
        encoded, _ = self.eva(x)

        # Project back to output shape
        decoded = rearrange(encoded, "b (w h d) c -> b c w h d", w=W, h=H, d=D)
        decoded = self.up_projection(decoded)

        return decoded
