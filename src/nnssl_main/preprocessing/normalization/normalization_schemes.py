from enum import Enum
from typing import Optional, Type
import numpy as np


class NormalizationScheme(Enum):
    Z_SCORE = "ZScoreNormalization"
    NO_NORMALIZATION = "NoNormalization"
    RESCALE_TO_01 = "RescaleTo01"
    RGB_TO_01 = "RGBTo01"


import numpy as np
from typing import Type, Protocol
from numbers import Number


class NormalizationProtocol(Protocol):
    def __call__(
        image: np.ndarray,
        target_dtype: Type[Number] = np.float32,
        use_mask_for_norm: Optional[bool] = None,
        non_zero_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        ...


def assert_bool_or_none(value):
    assert value is None or isinstance(value, bool)


def assert_min_max_for_rgb(image: np.ndarray):
    assert image.min() >= 0, "Your images do not seem to be RGB images"
    assert image.max() <= 255, "Your images do not seem to be RGB images"


def convert_dtype(image: np.ndarray, target_dtype: Type[Number]) -> np.ndarray:
    return image.astype(target_dtype)


def z_score_normalization(
    image: np.ndarray, use_mask_for_norm: bool, non_zero_mask: np.ndarray, target_dtype: Type[Number]
) -> np.ndarray:
    image = convert_dtype(image, target_dtype)
    if use_mask_for_norm:
        mask = non_zero_mask >= 0
        mean = image[mask].mean()
        std = image[mask].std()
        image[mask] = (image[mask] - mean) / max(std, 1e-8)
    else:
        mean = image.mean()
        std = image.std()
        image = (image - mean) / max(std, 1e-8)
    return image


def no_normalization(image: np.ndarray, target_dtype: Type[Number]) -> np.ndarray:
    return convert_dtype(image, target_dtype)


def rescale_to_01_normalization(image: np.ndarray, target_dtype: Type[Number]) -> np.ndarray:
    image = convert_dtype(image, target_dtype)
    image = image - image.min()
    return image / np.clip(image.max(), a_min=1e-8, a_max=None)


def rgb_to_01_normalization(image: np.ndarray, target_dtype: Type[Number]) -> np.ndarray:
    assert_min_max_for_rgb(image)
    image = convert_dtype(image, target_dtype)
    return image / 255.0



def apply_normalization(
    scheme: NormalizationScheme | str,
    image: np.ndarray,
    target_dtype: Type[Number] = np.float32,
    use_mask_for_norm: Optional[bool] = None,
    non_zero_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if isinstance(scheme, str):
        scheme = NormalizationScheme(scheme)
    if scheme == NormalizationScheme.Z_SCORE:
        return z_score_normalization(image, use_mask_for_norm, non_zero_mask, target_dtype)
    elif scheme == NormalizationScheme.NO_NORMALIZATION:
        return no_normalization(image, target_dtype)
    elif scheme == NormalizationScheme.RESCALE_TO_01:
        return rescale_to_01_normalization(image, target_dtype)
    elif scheme == NormalizationScheme.RGB_TO_01:
        return rgb_to_01_normalization(image, target_dtype)
    else:
        raise ValueError("Unknown normalization scheme")
