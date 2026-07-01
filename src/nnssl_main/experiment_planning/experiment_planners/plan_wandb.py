from dataclasses import dataclass, asdict, is_dataclass
from typing import Any
from nnssl.experiment_planning.experiment_planners.plan import Plan, ConfigurationPlan
import json
import numpy as np

def dataclass_to_dict(data):
    if is_dataclass(data):
        return {k: dataclass_to_dict(v) for k, v in asdict(data).items()}
    else:
        return data


@dataclass
class ConfigurationPlan_wandb(ConfigurationPlan):
    data_identifier: str
    preprocessor_name: str
    batch_size: int
    batch_dice: bool
    patch_size: np.ndarray
    median_image_size_in_voxels: np.ndarray
    spacing: np.ndarray
    normalization_schemes: list[str]
    use_mask_for_norm: list[str]
    UNet_class_name: str
    UNet_base_num_features: int
    n_conv_per_stage_encoder: tuple[int]
    n_conv_per_stage_decoder: tuple[int]
    num_pool_per_axis: list[int]
    pool_op_kernel_sizes: list[list[int]]
    conv_kernel_sizes: list[list[int]]
    unet_max_num_features: int
    resampling_fn_data: str
    resampling_fn_data_kwargs: dict[str, Any]
    resampling_fn_mask: str
    resampling_fn_mask_kwargs: dict[str, Any]
    mask_ratio: float=None
    vit_patch_size: list[int]=None
    embed_dim: int=None
    encoder_eva_depth: int=None
    encoder_eva_numheads: int=None
    decoder_eva_depth: int=None
    decoder_eva_numheads: int=None
    initial_lr: float=None,

    #
    #
    # def __getitem__(self, key):
    #     return getattr(self, key)
    #
    # def __setitem__(self, key, value):
    #     setattr(self, key, value)
    #
    # def __delitem__(self, key):
    #     delattr(self, key)
    #
    # def __contains__(self, key):
    #     return hasattr(self, key)
    #
    # def __len__(self):
    #     return len(self.__dict__)
    #
    # def keys(self):
    #     return self.__dict__.keys()
    #
    # def values(self):
    #     return [getattr(self, key) for key in self.keys()]
    #
    # def items(self):
    #     return [(key, getattr(self, key)) for key in self.keys()]


@dataclass
class Plan_wandb(Plan):
    dataset_name: str
    plans_name: str
    original_median_spacing_after_transp: list[float]
    original_median_shape_after_transp: list[int]
    image_reader_writer: str
    transpose_forward: list[int]
    transpose_backward: list[int]
    configurations: dict[str, ConfigurationPlan]
    experiment_planner_used: str

    # def __getitem__(self, key):
    #     return getattr(self, key)
    #
    # def __setitem__(self, key, value):
    #     setattr(self, key, value)
    #
    # def __delitem__(self, key):
    #     delattr(self, key)
    #
    # def __contains__(self, key):
    #     return hasattr(self, key)
    #
    # def _expected_save_directory(self):
    #     pp_path = os.environ.get("nnssl_preprocessed")
    #     if pp_path is None:
    #         raise RuntimeError(
    #             "nnssl_preprocessed environment variable not set. This is where the preprocessed data will be saved."
    #         )
    #     return os.path.join(pp_path, self.dataset_name, self.plans_name + ".json")
    #
    # def save_to_file(self, overwrite=False):
    #     save_dir = self._expected_save_directory()
    #     print(f"Saving plan to {save_dir}...")
    #     if os.path.isfile(save_dir) and not overwrite:
    #         return
    #     os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    #     with open(save_dir, "w") as f:
    #         json.dump(self._json_serializable(), f, indent=4, sort_keys=False)
    #
    # def _json_serializable(self) -> dict:
    #     only_dicts = dataclass_to_dict(self)
    #     recursive_fix_for_json_export(only_dicts)
    #     return only_dicts
    #
    # def __len__(self):
    #     return len(self.__dict__)
    #
    # def keys(self):
    #     return self.__dict__.keys()
    #
    # def values(self):
    #     return [getattr(self, key) for key in self.keys()]
    #
    # def items(self):
    #     return [(key, getattr(self, key)) for key in self.keys()]
    #
    @staticmethod
    def load_from_file(path: str):
        json_dict: dict = json.load(open(path, "r"))
        configs = {k: ConfigurationPlan_wandb(**v) for k, v in json_dict["configurations"].items()}
        json_dict["configurations"] = configs
        return Plan(**json_dict)
    #
    # def image_reader_writer_class(self) -> "Type[BaseReaderWriter]":
    #     return recursive_find_reader_writer_by_name(self.image_reader_writer)
