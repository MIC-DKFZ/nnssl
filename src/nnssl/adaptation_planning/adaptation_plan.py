from dataclasses import dataclass
from typing import Any, Literal
import numpy as np

ARCHITECTURE_PRESETS = Literal[
    "ResEncL",
    "NoSkipResEncL",
    "PrimusS",
    "PrimusB",
    "PrimusM",
    "PrimusL",
    "ResidualEncoderUNet",
    "PlainConvUNet",
]


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
        else:
            serialized_kwargs[key] = value
    return serialized_kwargs


@dataclass
class AdaptationPlan:
    architecture_name: ARCHITECTURE_PRESETS
    num_input_channels: int
    input_patch_size: tuple[int, int, int]
    state_dict_key_to_encoder: str
    state_dict_key_to_stem: str
    architecture_kwargs: dict[str, Any] = None

    def serialize(self):
        return {
            "architecture_name": self.architecture_name,
            "num_input_channels": self.num_input_channels,
            "input_patch_size": self.input_patch_size,
            "state_dict_key_to_encoder": self.state_dict_key_to_encoder,
            "state_dict_key_to_stem": self.state_dict_key_to_stem,
            "architecture_kwargs": serialize_kwargs(self.architecture_kwargs) if self.architecture_kwargs else None,
        }
