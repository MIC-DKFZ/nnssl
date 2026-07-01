from dataclasses import dataclass, asdict, is_dataclass
from dataclasses import field
from typing import Any, Literal, Sequence, Type, get_args
import numpy as np
from torch import nn
import io
import json

from nnssl.architectures.get_network_from_plan import get_network_from_plans
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan


ARCHITECTURE_PRESETS: Type[str] = Literal[
    "ResEncL",
    "NoSkipResEncL",
    "PrimusS",
    "PrimusB",
    "PrimusM",
    "PrimusL",
    "ResidualEncoderUNet",
    "PlainConvUNet",
]

DYN_ARCHITECTURE_PRESETS = Literal[
    "ResidualEncoderUNet",
    "PlainConvUNet",
]


def recursive_asdict(obj: dataclass) -> dict:
    """Recursively convert dataclass to dictionary."""
    if isinstance(obj, list):
        return [recursive_asdict(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_asdict(item) for item in obj)
    elif isinstance(obj, dict):
        return {key: recursive_asdict(value) for key, value in obj.items()}
    elif is_dataclass(obj):
        return {key: recursive_asdict(value) for key, value in asdict(obj).items()}
    else:
        return obj


def serialize_kwargs(arch_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Serialize architecture kwargs to a dictionary."""
    serialized_kwargs = {}
    for key, value in arch_kwargs.items():
        if isinstance(value, list):
            serialized_kwargs[key] = [int(v) if isinstance(v, float) and v.is_integer() else v for v in value]
        elif isinstance(value, float):
            serialized_kwargs[key] = int(value) if value.is_integer() else value
        elif isinstance(value, np.ndarray):
            serialized_kwargs[key] = value.tolist()
        elif isinstance(value, dict):
            serialized_kwargs[key] = serialize_kwargs(value)
        else:
            serialized_kwargs[key] = value
    return serialized_kwargs


@dataclass
class DynamicArchitecturePlans:
    n_stages: int
    features_per_stage: list[int]
    conv_op: Type[nn.Module]
    kernel_sizes: list[tuple[int, int, int]]
    strides: list[tuple[int, int, int]]
    n_blocks_per_stage: int
    n_conv_per_stage_decoder: int
    conv_bias: bool
    norm_op: Type[nn.Module]
    norm_op_kwargs: dict[str, Any]
    dropout_op: Type[nn.Module] | None
    dropout_op_kwargs: dict[str, Any] | None
    nonlin: Type[nn.Module]
    nonlin_kwargs: dict[str, Any]
    # kwargs_requiring_import: list[str] = field(default_factory=lambda: ["conv_op", "norm_op", "nonlin", "dropout_op"])

    def serialize(self):
        serialized_arch_kwargs = asdict(self)
        for key in ["nonlin", "norm_op", "conv_op", "dropout_op"]:
            val = serialized_arch_kwargs[key]
            if val is not None and isinstance(val, type) and issubclass(val, nn.Module):
                serialized_arch_kwargs[key] = val.__module__ + "." + val.__name__
        return serialized_arch_kwargs

    def get_kwargs_requiring_import(self):
        kwargs_requiring_import = []
        for key, value in self.__dict__.items():
            if isinstance(value, type) and (issubclass(value, nn.Module)) or isinstance(value, str):
                kwargs_requiring_import.append(key)
        return kwargs_requiring_import


@dataclass
class ArchitecturePlans:
    arch_class_name: ARCHITECTURE_PRESETS
    arch_kwargs: DynamicArchitecturePlans | None = None
    arch_kwargs_requiring_import: list[str] | None = field(init=False, default=None)

    def __post_init__(self):
        if self.arch_kwargs:
            self.arch_kwargs_requiring_import = self.arch_kwargs.get_kwargs_requiring_import()

    def serialize(self):
        serialized_arch_kwargs = self.arch_kwargs.serialize() if self.arch_kwargs else None
        return {
            "arch_class_name": self.arch_class_name,
            "arch_kwargs": serialized_arch_kwargs,
            "arch_kwargs_requiring_import": self.arch_kwargs_requiring_import,
        }


@dataclass
class AdaptationPlan:
    """
    Datastructure to provide all details necessary to properly adapt the model to a downstream dataset.
    :param arch_class_name: Name of the architecture class to be used.
    :param pretrain_plan: Pre-training plan.
    :param pretrain_num_input_channels: Number of input channels of the pre-trained model.
    :param key_to_encoder: Key to the encoder in the state dict.
    :param key_to_stem: Key to the stem in the state dict.
    """

    architecture_plans: ArchitecturePlans
    pretrain_plan: Plan
    pretrain_num_input_channels: int
    recommended_downstream_patchsize: tuple[int, int, int]
    key_to_encoder: str
    key_to_stem: str
    keys_to_in_proj: Sequence[str]
    key_to_lpe: str | None = None

    def serialize(self):
        serialized_plan = asdict(self)
        serialized_plan["architecture_plans"] = self.architecture_plans.serialize()
        serialized_plan["pretrain_plan"] = self.pretrain_plan.serialize()
        return serialize_kwargs(serialized_plan)

    @staticmethod
    def from_dict(data: dict):
        architecture_plans = ArchitecturePlans(
            arch_class_name=data["architecture_plans"]["arch_class_name"],
            arch_kwargs=(
                DynamicArchitecturePlans(**data["architecture_plans"]["arch_kwargs"])
                if data["architecture_plans"]["arch_kwargs"]
                else None
            ),
        )
        pretrain_plan = Plan.from_dict(data["pretrain_plan"])
        pretrain_num_input_channels = data["pretrain_num_input_channels"]
        recommended_downstream_patchsize = data["recommended_downstream_patchsize"]
        key_to_encoder = data["key_to_encoder"]
        key_to_stem = data["key_to_stem"]
        keys_to_in_proj = data["keys_to_in_proj"]
        return AdaptationPlan(
            architecture_plans=architecture_plans,
            pretrain_plan=pretrain_plan,
            recommended_downstream_patchsize=recommended_downstream_patchsize,
            pretrain_num_input_channels=pretrain_num_input_channels,
            key_to_encoder=key_to_encoder,
            key_to_stem=key_to_stem,
            keys_to_in_proj=tuple(keys_to_in_proj),
            key_to_lpe=data.get("key_to_lpe", None),
        )


if __name__ == "__main__":
    # ----- Simulate how the AdaptationPlan would be used in a real scenario ----- #

    # Assume authors are trying pre-training the model below.
    arch_kwargs = DynamicArchitecturePlans(
        n_stages=6,
        features_per_stage=[32, 64, 128, 256, 320, 480],
        conv_op=nn.Conv3d,
        kernel_sizes=[(3, 3, 3) for _ in range(6)],
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        n_blocks_per_stage=[1, 3, 4, 6, 6, 8],
        n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        dropout_op=None,
        dropout_op_kwargs=None,
    )
    configuration_plan = ConfigurationPlan(
        data_identifier="nnsslPlans_onemmiso",
        preprocessor_name="DefaultPreprocessor",
        spacing_style="onemmiso",
        normalization_schemes=["ZScoreNormalization"],
        use_mask_for_norm=False,
        resampling_fn_data="resample_data_or_seg_to_shape",
        resampling_fn_data_kwargs={"is_seg": False, "order": 3, "order_z": 0, "force_separate_z": None},
        resampling_fn_mask="resample_data_or_seg_to_shape",
        resampling_fn_mask_kwargs={"is_seg": False, "order": 1, "order_z": 0, "force_separate_z": None},
        spacing=[1, 1, 1],
        patch_size=None,
    )

    plan = Plan(
        dataset_name="Dataset802_OpenNeuro",
        plans_name="nnsslPlans",
        original_median_spacing_after_transp=[1.0, 1.0, 1.0],  # Arbitrary
        image_reader_writer="NibabelReaderWriter",
        transpose_forward=[0, 1, 2],
        transpose_backward=[0, 1, 2],
        experiment_planner_used="ExperimentPlanner",
        configurations=configuration_plan,  # We just want to save the used configuration plan!
    )

    # Provide some more infos on the pre-training spacing and patch size
    # and the architecture name
    arch_plans = ArchitecturePlans(
        arch_class_name="ResidualEncoderUNet",
        arch_kwargs=arch_kwargs,
    )
    adaptation_plan = AdaptationPlan(
        architecture_plans=arch_plans,
        pretrain_plan=plan,
        pretrain_num_input_channels=1,
        recommended_downstream_patchsize=(160, 160, 160),
        key_to_encoder="encoder.stages",
        key_to_stem="encoder.stem",
        keys_to_in_proj=("encoder.stem.convs.0.conv", "encoder.stem.convs.0.all_modules.0"),
    )
    serialized_adaptation_plan = adaptation_plan.serialize()
    # Write the serialized adaptation plan to an IO buffer
    buffer = io.StringIO()
    json.dump(serialized_adaptation_plan, buffer)
    buffer.seek(0)  # Reset buffer position to the beginning

    # ------------- Simulate downstream usage of the adaptation plan ------------- #
    # First load from disk
    deserialized_adaptation_plan = json.load(buffer)

    NUM_INPUT_CHANNELS = 1
    NUM_OUTPUT_CHANNELS = 2

    print(deserialized_adaptation_plan)
    network = get_network_from_plans(
        arch_class_name=deserialized_adaptation_plan["architecture_plans"]["arch_class_name"],
        arch_kwargs=deserialized_adaptation_plan["architecture_plans"]["arch_kwargs"],
        arch_kwargs_req_import=deserialized_adaptation_plan["architecture_plans"]["arch_kwargs_requiring_import"],
        input_channels=1,  # Should be different
        output_channels=2,
        allow_init=False,  # Always false
        deep_supervision=False,
    )
    print(network)
