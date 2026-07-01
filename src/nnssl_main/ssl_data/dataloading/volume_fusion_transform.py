from typing import Sequence, Tuple

import matplotlib.pyplot as plt
from batchgenerators.transforms.abstract_transforms import AbstractTransform

import numpy as np


def _mix_image(
    foreground_image: np.ndarray, background_image: np.ndarray, mixing_coefficient: np.ndarray
) -> np.ndarray:
    """
    Mixes two images together using a mixing coefficient array

    :param foreground_image: The image to be mixed in the foreground
    :param background_image: The image to be mixed in the background
    :param mixing_coefficient: The mixing coefficient array of the same shape as the images (range of [0, 1])
    :return: The mixed image
    """
    return mixing_coefficient * foreground_image + (1 - mixing_coefficient) * background_image


def _get_bboxes_within_image_bounds(
    num_patches: int,
    patch_size: tuple[int, int, int],
    bbox_bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
):
    """
    Creates `num_patches` bounding boxes within the image bounds.
    Returns the bounding box dimensions and starting coordinates.

    :param num_patches: The number of bounding boxes to be created
    :param patch_size: The shape of the image
    :param bbox_bounds: The bounds of the bounding boxes in each dimension
    :return: The bounding box dimensions and starting coordinates
    """
    xs = np.random.randint(*bbox_bounds[0], size=num_patches)
    ys = np.random.randint(*bbox_bounds[1], size=num_patches)
    zs = np.random.randint(*bbox_bounds[2], size=num_patches)
    x_starts = np.random.randint(0, patch_size[0] - xs + 1, size=num_patches)
    y_starts = np.random.randint(0, patch_size[1] - ys + 1, size=num_patches)
    z_starts = np.random.randint(0, patch_size[2] - zs + 1, size=num_patches)

    return xs, ys, zs, x_starts, y_starts, z_starts


def _overlay_bbox(
    image: np.ndarray,
    values: tuple[float],
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    x_starts: np.ndarray,
    y_starts: np.ndarray,
    z_starts: np.ndarray,
):
    """
    Iterates over the bbox arrays, saves each bbox as dense array in the image.
    Later bbox will overwrite previous bbox if they overlap.
    Returns the dense image with the bboxes.
    """

    num_patches = len(xs)
    for idx in range(num_patches):
        x_start, x_size = x_starts[idx], xs[idx]
        y_start, y_size = y_starts[idx], ys[idx]
        z_start, z_size = z_starts[idx], zs[idx]
        image[:, x_start : x_start + x_size, y_start : y_start + y_size, z_start : z_start + z_size] = values[idx]
    return image


def mix_batch(
    images: np.ndarray,
    vf_subpatch_count: Tuple[int, int],
    vf_subpatch_size: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    vf_mixing_coefficients: Sequence[float],
):
    """
    Takes a batch of images and mixes them together using the Volume Fusion technique.

    :param images: The batch of images to be mixed
    :param vf_subpatch_count: The number of subpatches to be mixed [min, max]
    :param vf_subpatch_size: The size of the subpatches to be mixed [min, max] in Voxels for each dimension
    :param vf_mixing_coefficients: The mixing coefficients to be used for the mixing within [0, 1] range
    """
    # Split the batch into two halves
    vf_mixing_coefficients = np.array(vf_mixing_coefficients)
    batch_size = images.shape[0]
    half_batch = batch_size // 2
    foreground_images = images[:half_batch]
    background_images = images[half_batch:]
    _, _, X, Y, Z = foreground_images.shape

    alpha_images = np.zeros_like(foreground_images)
    masks = np.zeros((half_batch, 1, X, Y, Z), dtype=np.float32)

    for i in range(half_batch):
        num_patches = np.random.randint(*vf_subpatch_count)
        indices = np.random.randint(0, len(vf_mixing_coefficients), size=num_patches)
        alphas = vf_mixing_coefficients[indices]

        xs, ys, zs, x_starts, y_starts, z_starts = _get_bboxes_within_image_bounds(
            num_patches, (X, Y, Z), vf_subpatch_size
        )
        alpha_images[i] = _overlay_bbox(alpha_images[i], alphas, xs, ys, zs, x_starts, y_starts, z_starts)
        masks[i] = _overlay_bbox(masks[i], indices, xs, ys, zs, x_starts, y_starts, z_starts)

    mixed_images = _mix_image(foreground_images, background_images, alpha_images)
    return mixed_images, masks


class VolumeFusionTransform(AbstractTransform):

    def __init__(
        self,
        vf_mixing_coefficients: Sequence[float],
        vf_subpatch_count: Tuple[int, int],
        vf_subpatch_size: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
        data_key: str = "data",
    ):
        """
        The Volume Fusion Transform is a data augmentation technique that mixes two images together.
        The mixing is done by drawing M - the number of patches - a mixing factor α - the location of the patch

        returns the mixed image, and the mixing mask.

        :param vf_mixing_coefficients: The mixing coefficients to be used for the mixing within [0, 1] range
        :param vf_subpatch_count: The number of subpatches to be mixed [min, max]
        :param vf_subpatch_size: The size of the subpatches to be mixed [min, max] in Voxels for X Y Z dimension
        """

        self.data_key: str = data_key
        self.vf_mixing_coefficients: Sequence[float] = vf_mixing_coefficients
        self.vf_subpatch_count: Tuple[int, int] = vf_subpatch_count
        self.vf_subpatch_size: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] = vf_subpatch_size

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        if data is None:
            raise ValueError(f"No data found for key {self.data_key}")

        mixed_images, masks = mix_batch(
            data,
            vf_subpatch_count=self.vf_subpatch_count,
            vf_subpatch_size=self.vf_subpatch_size,
            vf_mixing_coefficients=self.vf_mixing_coefficients,
        )
        # depth_idx = mixed_images.shape[2]//2
        # for i in range(len(mixed_images)):
        #     plt.imsave(f"images/mixed_image_center_{i:02}.png", mixed_images[i, 0, depth_idx], cmap="gray")
        # exit()

        return {"input": mixed_images, "target": masks}
