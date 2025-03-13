from typing import Literal, Optional, Sequence
from torch import nn
from nnssl.architectures.backbones.primus import Primus
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

from nnssl.architectures.backbones.utils.nsUNet import ResidualEncoderUNet_noskip


SUPPORTED_ARCHITECTURES = Literal["ResEncL", "NoSkipResEncL" "PrimusS", "PrimusB", "PrimusM", "PrimusL"]
PRIMUS_SCALES = Literal["S", "M", "B", "L"]


def get_res_enc_l(
    num_input_channels: int, num_output_channels: int, deep_supervision: bool = False
) -> ResidualEncoderUNet:
    """
    Creates the ResEnc-L architecture used in "Revisiting MAE Pre-training ..."
    https://arxiv.org/abs/2410.23132
    """
    n_stages = 6
    network = ResidualEncoderUNet(
        input_channels=num_input_channels,
        n_stages=n_stages,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
        num_classes=num_output_channels,
        n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=deep_supervision,
    )
    return network


def get_noskip_res_enc_l(num_input_channels: int, num_output_channels: int) -> ResidualEncoderUNet:
    """
    Creates the ResEnc-L architecture used in "Revisiting MAE Pre-training ..."
    https://arxiv.org/abs/2410.23132
    """
    network = ResidualEncoderUNet_noskip(
        input_channels=num_input_channels,
        n_stages=6,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3] for _ in range(6)],
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
        num_classes=num_output_channels,
        n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
    )
    return network


def get_primus(
    scale: PRIMUS_SCALES,
    input_channels: int,
    output_channels: int,
    input_shape: Sequence[int, int, int],
    token_patch_size=(8, 8, 8),
    kwargs: Optional[dict] = {
        "drop_path_rate": 0.2,
        "init_values": 0.1,
        "scale_attn_inner": True,
        "decoder_at": nn.GELU,
    },
) -> Primus:
    """
    Allows creation of the Primus S/B/M/L architectures.
    https://arxiv.org/abs/2503.01835
    """
    if scale == "S":
        n_layers = 12
        n_head = 6
        embed_dim = 396
    elif scale == "B":
        n_layers = 12
        n_head = 12
        embed_dim = 792
    elif scale == "M":
        n_layers = 16
        n_head = 12
        embed_dim = 864
    elif scale == "L":
        n_layers = 24
        n_head = 16
        embed_dim = 1056

    model = Primus(
        input_channels=input_channels,
        embed_dim=embed_dim,
        patch_embed_size=token_patch_size,
        output_channels=output_channels,
        eva_depth=n_layers,
        eva_numheads=n_head,
        input_shape=input_shape,
        drop_path_rate=0.2,
        **kwargs,
    )
    return model
