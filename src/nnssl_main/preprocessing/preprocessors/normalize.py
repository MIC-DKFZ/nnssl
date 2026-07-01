
from nnssl.preprocessing.normalization.normalization_schemes import apply_normalization
import numpy as np


def normalize_arr(
    data: np.ndarray,
    non_zero_mask: np.ndarray,
    normalization_schemes: list[str],
    use_mask_for_norm: list[bool],
) -> np.ndarray:
    """
    This code can give you a stroke if you read it.
    Who initializes a new class everytime a function is called?
    (I did not write this, but I am sorry you got here and had to read this as well.)
    """
    for c in range(data.shape[0]):
        data[c] = apply_normalization(
            scheme=normalization_schemes[c],
            image=data[c],
            target_dtype=data.dtype,
            use_mask_for_norm=use_mask_for_norm[c],
            non_zero_mask=non_zero_mask[c],
        )
    return data

