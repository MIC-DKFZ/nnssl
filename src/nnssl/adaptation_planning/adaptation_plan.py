from dataclasses import dataclass
from typing import Literal

ARCHITECTURE_PRESETS = Literal["ResEncL", "NoSkipResEncL", "PrimusS", "PrimusB", "PrimusM", "PrimusL"]


@dataclass
class AdaptationPlan:
    architecture_name: ARCHITECTURE_PRESETS
    num_input_channels: int
    input_patch_size: tuple[int, int, int]
    state_dict_key_to_encoder: str
    state_dict_key_to_stem: str

    def serialize(self):
        return {
            "architecture_name": self.architecture_name,
            "num_input_channels": self.num_input_channels,
            "input_patch_size": self.input_patch_size,
            "state_dict_key_to_encoder": self.state_dict_key_to_encoder,
            "state_dict_key_to_stem": self.state_dict_key_to_stem,
        }
