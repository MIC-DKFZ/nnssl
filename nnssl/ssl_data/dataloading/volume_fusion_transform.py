from typing import Tuple
from batchgenerators.transforms.abstract_transforms import AbstractTransform

from einops import rearrange
import numpy as np
from loguru import logger
import torch


class VolumeFusionTransform(AbstractTransform):

    def __init__(
        self,
        vf_mixing_coefficients: np.ndarray,
        vf_subpatch_count: Tuple[int, int],
        vf_subpatch_size: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
        data_key="data",
    ):
        """
        The Volume Fusion Transform is a data augmentation technique that mixes two images together.
        The mixing is done by drawing M - the number of patches - a mixing factor α - the location of the patch

        returns the mixed image, and the mixing mask.
        """

        self.data_key = data_key
        self.vf_mixing_coefficients = vf_mixing_coefficients
        self.vf_subpatch_count = vf_subpatch_count
        self.vf_subpatch_size = vf_subpatch_size

    def mix_batch(self, images: np.ndarray):
        """
        Mixes the batch of images by drawing M - the number of patches - a mixing factor α - the location of the patch

        :return: mixed_images [b,c,x,y,z] , masks [b, 1, x, y, z] -- not one-hot encoded
        """
        # Split the batch into two halves
        batch_size = images.shape[0]
        half_batch = batch_size // 2
        foreground_images = images[:half_batch]
        background_images = images[half_batch:]
        _, _, D, H, W = foreground_images.shape

        alpha_images = np.zeros_like(foreground_images)
        masks = np.zeros((half_batch, 1, D, H, W), dtype=np.float32)

        # for i in range(half_batch):
        #     num_patches = np.random.randint(*self.vf_subpatch_count)
        #     indices = np.random.randint(0, len(self.vf_mixing_coefficients), size=num_patches)

        #     ds = np.random.randint(*self.vf_subpatch_size[0], size=num_patches)
        #     hs = np.random.randint(*self.vf_subpatch_size[1], size=num_patches)
        #     ws = np.random.randint(*self.vf_subpatch_size[2], size=num_patches)
        #     d_starts = np.random.randint(0, D - ds + 1, size=num_patches)
        #     h_starts = np.random.randint(0, H - hs + 1, size=num_patches)
        #     w_starts = np.random.randint(0, W - ws + 1, size=num_patches)
        #     alphas = self.vf_mixing_coefficients[indices]

        #     # Sequential application (can be modified for parallel execution if non-overlapping is guaranteed)
        #     for idx in range(num_patches):
        #         d_start, d_size = d_starts[idx], ds[idx]
        #         h_start, h_size = h_starts[idx], hs[idx]
        #         w_start, w_size = w_starts[idx], ws[idx]
        #         alpha_images[i, :, d_start:d_start+d_size, h_start:h_start+h_size, w_start:w_start+w_size] = alphas[idx]
        #         masks[i, d_start:d_start+d_size, h_start:h_start+h_size, w_start:w_start+w_size] = indices[idx]

        mixed_images = alpha_images * foreground_images + (1 - alpha_images) * background_images

        return mixed_images, masks

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        if data is None:
            raise ValueError(f"No data found for key {self.data_key}")

        mixed_images, masks = self.mix_batch(data)

        return {"input": mixed_images, "target": masks}
