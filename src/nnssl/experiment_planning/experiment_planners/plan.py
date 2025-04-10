from dataclasses import dataclass, asdict, is_dataclass
import os
from typing import Any, Literal, Type

import json
import numpy as np


from nnssl.imageio.reader_writer_registry import recursive_find_reader_writer_by_name
from nnssl.utilities.json_export import recursive_fix_for_json_export

PREPROCESS_SPACING_STYLES = Literal["onemmiso", "median", "noresample"]


def dataclass_to_dict(data):
    if is_dataclass(data):
        return {k: dataclass_to_dict(v) for k, v in asdict(data).items()}
    else:
        return data


@dataclass
class ConfigurationPlan:
    data_identifier: str
    preprocessor_name: str
    spacing_style: PREPROCESS_SPACING_STYLES
    normalization_schemes: list[str]
    use_mask_for_norm: list[str]
    resampling_fn_data: str
    resampling_fn_data_kwargs: dict[str, Any]
    resampling_fn_mask: str
    resampling_fn_mask_kwargs: dict[str, Any]
    spacing: np.ndarray | list[float] | None = None
    patch_size: np.ndarray | list[float] | None = None

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        delattr(self, key)

    def __contains__(self, key):
        return hasattr(self, key)

    def __len__(self):
        return len(self.__dict__)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return [getattr(self, key) for key in self.keys()]

    def items(self):
        return [(key, getattr(self, key)) for key in self.keys()]

    def serialize(self) -> dict:
        if self.spacing_style != "noresample":
            assert (
                self.spacing is not None
            ), "Spacing is None in the ConfigurationPlan. Please set in init for reproducibility!"
        assert (
            self.patch_size is not None
        ), "Patch size is None in the ConfigurationPlan. Please set in init for reproducibility!"
        for norm in self.normalization_schemes:
            assert norm is not None, (
                "At least one Normalization scheme is None in the ConfigurationPlan."
                + f"Please set in init for reproducibility!\n Current value: {self.normalization_schemes}"
            )
        return asdict(self)


@dataclass
class Plan:
    dataset_name: str
    plans_name: str
    original_median_spacing_after_transp: list[float]
    image_reader_writer: str
    transpose_forward: list[int]
    transpose_backward: list[int]
    configurations: dict[str, ConfigurationPlan]
    experiment_planner_used: str

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        delattr(self, key)

    def __contains__(self, key):
        return hasattr(self, key)

    def _expected_save_directory(self):
        pp_path = os.environ.get("nnssl_preprocessed")
        if pp_path is None:
            raise RuntimeError(
                "nnssl_preprocessed environment variable not set. This is where the preprocessed data will be saved."
            )
        return os.path.join(pp_path, self.dataset_name, self.plans_name + ".json")

    def save_to_file(self, overwrite=False):
        save_dir = self._expected_save_directory()
        print(f"Saving plan to {save_dir}...")
        if os.path.isfile(save_dir) and not overwrite:
            return
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        with open(save_dir, "w") as f:
            json.dump(self._json_serializable(), f, indent=4, sort_keys=False)

    def _json_serializable(self) -> dict:
        only_dicts = dataclass_to_dict(self)
        recursive_fix_for_json_export(only_dicts)
        return only_dicts

    def __len__(self):
        return len(self.__dict__)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return [getattr(self, key) for key in self.keys()]

    def items(self):
        return [(key, getattr(self, key)) for key in self.keys()]

    @staticmethod
    def load_from_file(path: str):
        json_dict: dict = json.load(open(path, "r"))
        return Plan.from_dict(json_dict)

    @staticmethod
    def from_dict(json_dict: dict):
        configs = {k: ConfigurationPlan(**v) for k, v in json_dict["configurations"].items()}

        json_dict["configurations"] = configs
        return Plan(**json_dict)

    def image_reader_writer_class(self) -> "Type[BaseReaderWriter]":
        return recursive_find_reader_writer_by_name(self.image_reader_writer)

    def serialize(self) -> dict:
        res = asdict(self)
        res["configurations"] = {k: v.serialize() for k, v in self.configurations.items()}
        return res
