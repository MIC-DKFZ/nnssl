from typing import Literal
from dynamic_network_architectures.architectures.abstract_arch import AbstractDynamicNetworkArchitectures
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.architectures.primus import PrimusS, PrimusB, PrimusM, PrimusL
from torch import nn
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from nnssl.architectures.architecture_registry import (
    SUPPORTED_ARCHITECTURES,
    get_res_enc_l,
    get_noskip_res_enc_l,
)
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan


def get_network_by_name(
    configuration_plan: ConfigurationPlan,
    architecture_name: SUPPORTED_ARCHITECTURES,
    num_input_channels: int,
    num_output_channels: int,
    encoder_only: bool = False,
    deep_supervision: bool = False,
    arch_kwargs: dict | None = None,
) -> AbstractDynamicNetworkArchitectures:
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than inferring it again from the plans here.
    """
    if architecture_name == "ResEncL":
        model = get_res_enc_l(num_input_channels, num_output_channels, deep_supervision)
    elif architecture_name == "NoSkipResEncL":
        model = get_noskip_res_enc_l(num_input_channels, num_output_channels)
    elif architecture_name in ["PrimusS", "PrimusB", "PrimusM", "PrimusL"]:
        if architecture_name == "PrimusS":
            model = PrimusS(
                input_channels=num_input_channels,
                output_channels=num_output_channels,
                input_shape=configuration_plan.patch_size,
                patch_embed_size=(8, 8, 8),
            )
        elif architecture_name == "PrimusB":
            model = PrimusB(
                input_channels=num_input_channels,
                output_channels=num_output_channels,
                input_shape=configuration_plan.patch_size,
                patch_embed_size=(8, 8, 8),
            )
        elif architecture_name == "PrimusM":
            model = PrimusM(
                input_channels=num_input_channels,
                output_channels=num_output_channels,
                input_shape=configuration_plan.patch_size,
                patch_embed_size=(8, 8, 8),
            )
        elif architecture_name == "PrimusL":
            model = PrimusL(
                input_channels=num_input_channels,
                output_channels=num_output_channels,
                input_shape=configuration_plan.patch_size,
                patch_embed_size=(8, 8, 8),
            )
        else:
            raise ValueError(f"Architecture {architecture_name} is not supported.")
    else:
        raise ValueError(f"Architecture {architecture_name} is not supported.")

    if encoder_only:
        if architecture_name in ["ResEncL", "NoSkipResEncL"]:
            model: ResidualEncoderUNet
            try:
                model = model.encoder
                model.key_to_encoder = model.key_to_encoder.replace("encoder.", "")
                model.keys_to_in_proj = [k.replace("encoder.", "") for k in model.keys_to_in_proj]
            except AttributeError:
                raise RuntimeError("Trying to get the 'encoder' of the network failed. Cannot return encoder only.")
        elif architecture_name in ["PrimusS", "PrimusB", "PrimusM", "PrimusL"]:
            raise NotImplementedError("Cannot return encoder only for Primus architectures.")
    return model
