from typing import Tuple
import numpy as np
import torch
from nnssl.training.nnsslTrainer.evaMAE.dynamic_network.helper import convert_dim_to_conv_op
from einops import rearrange
from torch import nn


class LayerNormNd(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[None, :, *tuple([None] * (x.ndim - 2))] * x + self.bias[None, :, *tuple([None] * (x.ndim - 2))]
        return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    Loosely inspired by https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L364

    """

    def __init__(
            self,
            patch_size: Tuple[int, ...] = (16, 16, 16),
            input_channels: int = 3,
            embed_dim: int = 768,
    ) -> None:
        """
        Args:
            patch_size (Tuple): patch size.
            padding (Tuple): padding size of the projection layer.
            input_channels (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = convert_dim_to_conv_op(len(patch_size))(
            input_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        returns shape (B, embed_dim, px, py, pz) where (px, py, pz) is patch_size.
        This output will need to be rearranged to whatever your transformer expects!
        """
        x = self.proj(x)
        return x


class PatchDecode(nn.Module):
    """
    Loosely inspired by SAM decoder
    https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py#L53
    """

    def __init__(
            self,
            patch_size,
            embed_dim: int,
            out_channels: int,
            norm=LayerNormNd,
            activation=nn.GELU,
    ):
        """
        patch size must be 2^x, so 2, 4, 8, 16, 32, etc. Otherwise we die
        """
        super().__init__()

        def _round_to_8(inp):
            return int(max(8, np.round((inp + 1e-6) / 8) * 8))

        num_stages = int(np.log(max(patch_size)) / np.log(2))
        strides = [[2 if (p / 2**n) % 2 == 0 else 1 for p in patch_size] for n in range(num_stages)][::-1]
        dim_red = (embed_dim / (2 * out_channels)) ** (1/num_stages)

        # don't question me
        channels = [embed_dim] + [_round_to_8(embed_dim / dim_red ** (x + 1)) for x in range(num_stages)]
        channels[-1] = out_channels

        stages = []
        for s in range(num_stages - 1):
            stages.append(nn.Sequential(
                nn.ConvTranspose3d(channels[s], channels[s+1], kernel_size=strides[s], stride=strides[s]),
                norm(channels[s+1]),
                activation()
            ))
        stages.append(nn.ConvTranspose3d(channels[-2], channels[-1], kernel_size=strides[-1], stride=strides[-1]))
        self.decode = nn.Sequential(*stages)

    def forward(self, x):
        """
        Expects input of shape (B, embed_dim, px, py, pz)! This will require you to reshape the output of your transformer!
        """
        return self.decode(x)


if __name__ == '__main__':
    patch_size = (8, 8, 8)
    image = torch.zeros((1, 2, 32, 64, 24))
    embed_dim = 864

    patch_embed = PatchEmbed(patch_size, image.shape[1], embed_dim)
    patch_decode = PatchDecode(patch_size, embed_dim, 4)

    # this is how you would apply a transformer (at least for my implementation, you axis ordering might differ)!
    x = patch_embed(image)  # -> [1, 768, 5, 4, 3]
    B, C, W, H, D = x.shape
    x = rearrange(x, 'b c w h d -> b (h w d) c')  # -> [1, 60, 768]
    ######################
    # x = MY_TRANSFORMER(x)
    ######################
    x = rearrange(x, 'b (h w d) c -> b c w h d', h=H, w=W, d=D)  # -> [1, 768, 5, 4, 3]
    x = patch_decode(x)  # -> [1, 4, 20, 64, 24]

    assert all([i ==j for i, j in zip(x.shape[2:], image.shape[2:])])