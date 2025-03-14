from typing import Literal
from torch import nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

from nnssl.architectures.noskipResEncUNet import ResidualEncoderUNet_noskip


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
