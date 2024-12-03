from typing import Tuple

import torch
from nnssl.training.nnsslTrainer.evaMAE.dynamic_network.eva import Eva
from nnssl.training.nnsslTrainer.evaMAE.dynamic_network.vit_embed_decode import PatchEmbed, PatchDecode, LayerNormNd
from nnssl.training.nnsslTrainer.evaMAE.dynamic_network.weight_init import InitWeights_He
from einops import rearrange
from timm.layers import RotaryEmbeddingCat
from torch import nn

class EvaMAE(nn.Module):
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
                 use_abs_pos_emb: bool = True,
                 mlp_ratio=4 * 2 / 3,
                 drop_path_rate=0,
                 drop_path_scale: bool = True,
                 patch_drop_rate: float = 0.5,
                 proj_drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 rope_impl=RotaryEmbeddingCat,
                 rope_kwargs=None):
        """
        Masked Autoencoder with EVA attention-based encoder and decoder.
        """
        assert input_shape is not None
        assert len(input_shape) == 3, "Currently only 3D is supported"
        assert all([j % i == 0 for i, j in zip(patch_embed_size, input_shape)])

        super().__init__()

        # Patch embedding for encoder
        self.encoder_embed = PatchEmbed(patch_embed_size, input_channels, embed_dim)

        # Encoder using EVA
        self.encoder = Eva(
            embed_dim=embed_dim,
            depth=eva_depth,
            num_heads=eva_numheads,
            ref_feat_shape=tuple([i // ds for i, ds in zip(input_shape, patch_embed_size)]),
            num_reg_tokens=num_register_tokens,
            use_rot_pos_emb=use_rot_pos_emb,
            use_abs_pos_emb=use_abs_pos_emb,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            drop_path_scale=drop_path_scale,
            patch_drop_rate=patch_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            rope_impl=rope_impl,
            rope_kwargs=rope_kwargs
        )

        # Patch embedding for decoder
        self.decoder_embed = PatchDecode(patch_embed_size, embed_dim, output_channels,
                                         norm=decoder_norm,
                                         activation=decoder_act)

        # Decoder using EVA
        self.decoder = Eva(
            embed_dim=embed_dim,
            depth=eva_depth,
            num_heads=eva_numheads,
            ref_feat_shape=tuple([i // ds for i, ds in zip(input_shape, patch_embed_size)]),
            num_reg_tokens=num_register_tokens,
            use_rot_pos_emb=use_rot_pos_emb,
            use_abs_pos_emb=use_abs_pos_emb,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            drop_path_scale=drop_path_scale,
            patch_drop_rate=0, # No drop in the decoder
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            rope_impl=rope_impl,
            rope_kwargs=rope_kwargs
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=1e-6)

        self.encoder_embed.apply(InitWeights_He(1e-2))
        self.decoder_embed.apply(InitWeights_He(1e-2))

    def restore_full_sequence(self, x, keep_indices, num_patches):
        """
        Restore the full sequence by filling blanks with mask tokens and reordering.
        """
        
        B, num_kept, C = x.shape
        device = x.device

        # Create mask tokens for missing patches
        num_masked = num_patches - num_kept
        mask_tokens = self.mask_token.repeat(B, num_masked, 1)

        # Prepare an empty tensor for the restored sequence
        restored = torch.zeros(B, num_patches, C, device=device)

        # Assign the kept patches and mask tokens in the correct positions
        for i in range(B):
            kept_pos = keep_indices[i]
            masked_pos = torch.tensor([j for j in range(num_patches) if j not in kept_pos], device=device)
            restored[i, kept_pos] = x[i]
            restored[i, masked_pos] = mask_tokens[i, :len(masked_pos)]

        return restored

    def forward(self, x):
        # Encode patches
        x = self.encoder_embed(x)
        B, C, W, H, D = x.shape
        x = rearrange(x, 'b c w h d -> b (h w d) c')

        # Encode using EVA (internally applies masking with patch_drop_rate)
        encoded, keep_indices = self.encoder(x)

        # Restore full sequence with mask tokens
        num_patches = W * H * D
        restored_x = self.restore_full_sequence(encoded, keep_indices, num_patches)

        # Decode with restored sequence and rope embeddings
        decoded, _ = self.decoder(restored_x)

        # Project back to output shape
        decoded = rearrange(decoded, 'b (h w d) c -> b c w h d', h=W, w=H, d=D)
        decoded = self.decoder_embed(decoded)

        return decoded, keep_indices

if __name__ == "__main__":
    # Toy example for testing
    input_shape = (64, 64, 64)
    patch_embed_size = (8, 8, 8)
    model = EvaMAE(
        input_channels=3,
        embed_dim=192,
        patch_embed_size=patch_embed_size,
        output_channels=3,
        input_shape=input_shape,
        eva_depth=6,
        eva_numheads=8
    )

    # Random input tensor
    x = torch.rand((2, 3, *input_shape))  # Batch size 2

    # Forward pass
    output, keep_indices = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
