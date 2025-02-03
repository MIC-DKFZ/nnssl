from typing import Literal, Tuple
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.spatial_transforms import Rot90Transform, MirrorTransform, SpatialTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RenameTransform, NumpyToTensor
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform,
    ContrastAugmentationTransform,
    GammaTransform,
)

from einops import rearrange
import numpy as np
from loguru import logger
import torch


class VocoTransform(AbstractTransform):
    def __init__(
        self,
        voco_base_crop_count: Tuple[int, int, int],
        voco_crop_size: Tuple[int, int, int],
        aug: Literal["train", "none"] = "train",
        voco_target_crop_count: int = 4,
        data_key="data",
    ):
        """
        The VoCo Transform takes a standard crop that is post all augmentations and splits it into multiple smaller crops.
        The base crops are non-overlapping and form the basis for comparison with the target crops.
        The target crop(s) are partially overlapping with the base crops.
        The overlap degree between the two needs to be returned, as it is the training signal.

        Does not return the original crop anymore!
        return "base_crops", "target_crops", "base_target_crop_overlaps"
        """

        self.data_key = data_key
        self.voco_base_crop_count = voco_base_crop_count
        self.voco_crop_size = voco_crop_size
        self.voco_target_crop_count = voco_target_crop_count
        self.bounding_boxes = []
        self.aug: str = aug
        for i in range(self.voco_base_crop_count[0]):
            for j in range(self.voco_base_crop_count[1]):
                for k in range(self.voco_base_crop_count[2]):
                    self.bounding_boxes.append(
                        (
                            i * self.voco_crop_size[0],
                            j * self.voco_crop_size[1],
                            k * self.voco_crop_size[2],
                            (i + 1) * self.voco_crop_size[0],
                            (j + 1) * self.voco_crop_size[1],
                            (k + 1) * self.voco_crop_size[2],
                        )
                    )
        # Normally we would like to know which axes have the same spacing -- Otherwise rotating the crop will not work.
        #   However since our current plan involves leaving spacing ISO we just add this warning to not forget it.
        logger.warning(
            "Assuming that the axes have the same spacing. If not, the crop rotations will introduce changing spacings."
        )

        if aug == "train":
            self.crop_augmentations: Compose = Compose(
                [
                    GaussianNoiseTransform(p_per_sample=0.1),  # We just rotate in the axial plane
                    GaussianBlurTransform(
                        (0.5, 1.0), different_sigma_per_channel=True, p_per_sample=0.2, p_per_channel=0.5
                    ),
                    BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15),
                    ContrastAugmentationTransform(p_per_sample=0.15),
                    SimulateLowResolutionTransform(
                        zoom_range=(0.5, 1),
                        per_channel=True,
                        p_per_channel=0.5,
                        order_downsample=0,
                        order_upsample=3,
                        p_per_sample=0.1,
                        ignore_axes=None,
                    ),
                    GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1),
                    GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3),
                    MirrorTransform(axes=(0,)),
                    MirrorTransform(axes=(1,)),
                    MirrorTransform(axes=(2,)),
                    Rot90Transform(axes=(0, 1)),
                ]
            )

    def get_base_crops(self, data):
        """
        Splits the data into base crops.
        Returns all crops.

        :param data: [B, C, X, Y, Z] data to split into base crops.
        :return: [B, N_subcrops, C, X_subcrop, Y_subcrop, Z_subcrop] base crops
        """
        base_crops = []
        for i in range(self.voco_base_crop_count[0]):
            for j in range(self.voco_base_crop_count[1]):
                for k in range(self.voco_base_crop_count[2]):
                    crop = data[
                        :,
                        :,
                        i * self.voco_crop_size[0] : (i + 1) * self.voco_crop_size[0],
                        j * self.voco_crop_size[1] : (j + 1) * self.voco_crop_size[1],
                        k * self.voco_crop_size[2] : (k + 1) * self.voco_crop_size[2],
                    ]
                    base_crops.append(crop)
        return np.stack(base_crops, axis=1)


    def get_target_crops(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Defines a random crop that is partially overlapping with some of the base crops.
        :return: [B, N_subcrops, C, X_subcrop, Y_subcrop, Z_subcrop], overlaps [N_target_crop, N_base_crop]
        """

        image_wise_crop = []
        image_wise_overlaps = []

        crop_size = self.voco_crop_size
        total_volume = crop_size[0] * crop_size[1] * crop_size[2]

        # For each image in batch -- data shape: [B, C, X, Y, Z]
        for big_crop in data:
            target_crops, target_overlaps = [], []
            for _ in range(self.voco_target_crop_count):
                x_offset = np.random.randint(0, (big_crop.shape[1] - crop_size[0]) + 1)
                y_offset = np.random.randint(0, (big_crop.shape[2] - crop_size[1]) + 1)
                z_offset = np.random.randint(0, (big_crop.shape[3] - crop_size[2]) + 1)

                crop = big_crop[
                    :,
                    x_offset : x_offset + crop_size[0],
                    y_offset : y_offset + crop_size[1],
                    z_offset : z_offset + crop_size[2],
                ]
                target_crops.append(crop)

                # Calculate overlap with base crops
                target_base_crop_overlaps = []
                for bbox in self.bounding_boxes:
                    overlap_x = max(0, min(x_offset + crop_size[0], bbox[3]) - max(x_offset, bbox[0]))
                    overlap_y = max(0, min(y_offset + crop_size[1], bbox[4]) - max(y_offset, bbox[1]))
                    overlap_z = max(0, min(z_offset + crop_size[2], bbox[5]) - max(z_offset, bbox[2]))
                    overlap_volume = overlap_x * overlap_y * overlap_z
                    overlap_ratio = overlap_volume / total_volume
                    target_base_crop_overlaps.append(overlap_ratio)
                target_overlaps.append(np.array(target_base_crop_overlaps))
            image_wise_crop.append(np.stack(target_crops, axis=0))
            image_wise_overlaps.append(np.stack(target_overlaps, axis=0))  # [N_target_subcrops, N_base_subcrops]

        return np.stack(image_wise_crop, axis=0), np.stack(image_wise_overlaps, axis=0)

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        if data is None:
            raise ValueError(f"No data found for key {self.data_key}")

        base_crops = self.get_base_crops(data)  # [B, N_base_subcrops, C, X_subcrop, Y_subcrop, Z_subcrop]
        target_crops, gt_overlap = self.get_target_crops(data)
        # target_crops: [B, N_target_subcrops, C, X_subcrop, Y_subcrop, Z_subcrop]
        # gt_overlap: [B, N_target_subcrops, N_base_subcrops]
        B = base_crops.shape[0]
        if self.aug == "train":
            base_crops = rearrange(base_crops, "b n c x y z  -> (b n) c x y z")
            base_crops = self.crop_augmentations(**{"data": base_crops, "seg": None})["data"]
            base_crops = rearrange(base_crops, "(b n) c x y z -> b n c x y z", b=B)
            target_crops = rearrange(target_crops, "b n c x y z -> (b n) c x y z")
            target_crops = self.crop_augmentations(**{"data": target_crops, "seg": None})["data"]
            target_crops = rearrange(target_crops, "(b n) c x y z -> b n c x y z", b=B)

        base_crops_flat = rearrange(base_crops, "b n c x y z -> (b n) c x y z")
        target_crops_flat = rearrange(target_crops, "b n c x y z -> (b n) c x y z")
        joint_crops_flat = np.concatenate([base_crops_flat, target_crops_flat], axis=0)
        base_crop_index = base_crops_flat.shape[0]

        data_dict["all_crops"] = joint_crops_flat
        data_dict["base_crop_index"] = base_crop_index
        # data_dict["base_crops"] = base_crops
        # data_dict["target_crops"] = target_crops
        data_dict["base_target_crop_overlaps"] = gt_overlap
        return data_dict
