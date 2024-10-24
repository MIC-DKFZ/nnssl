from typing import Literal, Tuple
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.spatial_transforms import (
    Rot90Transform,
    MirrorTransform,
    SpatialTransform,
)
from batchgenerators.transforms.noise_transforms import (
    GaussianNoiseTransform,
    GaussianBlurTransform,
)
from batchgenerators.transforms.resample_transforms import (
    SimulateLowResolutionTransform,
)
from batchgenerators.transforms.spatial_transforms import (
    SpatialTransform,
    MirrorTransform,
)
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


class SimCLRTransform_deprecated(AbstractTransform):
    def __init__(self, transforms):
        """
        The SimCLR Transform takes the regular transforms and applies them to the same data twice.

        return tuple of transformed data_dicts
        """
        self.transforms = transforms
        self.rename = RenameTransform(in_key="data", out_key="image", delete_old=True)

    def __call__(self, **data_dict):
        renamed = self.rename(**data_dict)
        renamed["image"] = torch.from_numpy(renamed["image"]).squeeze().float()
        xi = self.transforms(**renamed)
        xj = self.transforms(**renamed)

        return {"image_i": xi["image"], "image_j": xj["image"]}


class SimCLRTransform(AbstractTransform):
    def __init__(
        self,
        crop_size: Tuple[int, int, int],
        aug: Literal["train", "none"] = "train",
        crop_count_per_image: int = 1,
        min_overlap_ratio: float = 0.5,
        data_key="data",
    ):
        """
        The SimCLR Transform takes a volume splits it into partially overlapping crops.
        The base crops are non-overlapping and form the basis for comparison with the target crops.
        The target crop(s) are partially overlapping with the base crops.
        The overlap degree between the two needs to be specified.

        Does not return the original volume anymore!
        return "base_crops", "target_crops"
        """

        self.data_key = data_key
        self.crop_count_per_image = crop_count_per_image
        self.crop_size = crop_size
        self.min_overlap_ratio = min_overlap_ratio

        self.min_overlap_per_axis = (
            int(crop_size[0] * min_overlap_ratio),
            int(crop_size[1] * min_overlap_ratio),
            int(crop_size[2] * min_overlap_ratio),
        )

        self.aug: str = aug
        # Normally we would like to know which axes have the same spacing -- Otherwise rotating the crop will not work.
        #   However since our current plan involves leaving spacing ISO we just add this warning to not forget it.
        logger.warning(
            "Assuming that the axes have the same spacing. If not, the crop rotations will introduce changing spacings."
        )

        if aug == "train":
            self.crop_augmentations: Compose = Compose(
                [
                    GaussianNoiseTransform(
                        p_per_sample=0.1
                    ),  # We just rotate in the axial plane
                    GaussianBlurTransform(
                        (0.5, 1.0),
                        different_sigma_per_channel=True,
                        p_per_sample=0.2,
                        p_per_channel=0.5,
                    ),
                    BrightnessMultiplicativeTransform(
                        multiplier_range=(0.75, 1.25), p_per_sample=0.15
                    ),
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
                    GammaTransform(
                        (0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1
                    ),
                    GammaTransform(
                        (0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3
                    ),
                    MirrorTransform(axes=(0,)),
                    MirrorTransform(axes=(1,)),
                    MirrorTransform(axes=(2,)),
                    Rot90Transform(axes=(0, 1)),
                ]
            )

    def get_reference_crops(self, data):
        """
        Splits the data into reference crops.
        Returns all crops and offsets so overlapping crops can be made.
        Note:
        In this form, multiple crops per image can from close proximity - this might lead to false negatives in the contrastive loss,
        but shouldn't be much of an issue.

        :param data: [B, C, X, Y, Z] data to split into base crops.
        :return: [B, N_CROPS_PER_IMAGE, C, X_subcrop, Y_subcrop, Z_subcrop] reference crops, [B, N_CROPS_PER_IMAGE, 3] reference offsets
        """
        offsets = np.zeros((data.shape[0], self.crop_count_per_image, 3))
        reference_crops = []
        for b in range(data.shape[0]):
            per_image_crops = []
            for i in range(self.crop_count_per_image):
                x_offset = np.random.randint(0, (data.shape[2] - self.crop_size[0]) + 1)
                y_offset = np.random.randint(0, (data.shape[3] - self.crop_size[1]) + 1)
                z_offset = np.random.randint(0, (data.shape[4] - self.crop_size[2]) + 1)
                crop = data[
                    b,
                    :,
                    x_offset : x_offset + self.crop_size[0],
                    y_offset : y_offset + self.crop_size[1],
                    z_offset : z_offset + self.crop_size[2],
                ]
                per_image_crops.append(crop)
                offsets[b, i, :] = [x_offset, y_offset, z_offset]
            reference_crops.append(np.stack(per_image_crops, axis=0))

        return np.stack(reference_crops, axis=0), offsets

    def get_overlapping_crops(self, data, reference_offsets):
        # Create overlapping crops while respecting array boundaries.
        overlapping_offsets = np.zeros((data.shape[0], self.crop_count_per_image, 3))
        overlapping_crops = []
        for b in range(data.shape[0]):
            per_image_crops = []
            for i in range(self.crop_count_per_image):
                x_offset_min = max(
                    0, reference_offsets[b, i, 0] - self.min_overlap_per_axis[0]
                )
                x_offset_max = min(
                    data.shape[2] - self.crop_size[0],
                    reference_offsets[b, i, 0] + self.min_overlap_per_axis[0],
                )

                y_offset_min = max(
                    0, reference_offsets[b, i, 1] - self.min_overlap_per_axis[1]
                )
                y_offset_max = min(
                    data.shape[3] - self.crop_size[1],
                    reference_offsets[b, i, 1] + self.min_overlap_per_axis[1],
                )

                z_offset_min = max(
                    0, reference_offsets[b, i, 2] - self.min_overlap_per_axis[2]
                )
                z_offset_max = min(
                    data.shape[4] - self.crop_size[2],
                    reference_offsets[b, i, 2] + self.min_overlap_per_axis[2],
                )

                # Sampling new offsets for each axis while ensuring the minimum overlap
                new_x_offset = np.random.randint(x_offset_min, x_offset_max + 1)
                new_y_offset = np.random.randint(y_offset_min, y_offset_max + 1)
                new_z_offset = np.random.randint(z_offset_min, z_offset_max + 1)

                crop = data[
                    b,
                    :,
                    new_x_offset : new_x_offset + self.crop_size[0],
                    new_y_offset : new_y_offset + self.crop_size[1],
                    new_z_offset : new_z_offset + self.crop_size[2],
                ]
                per_image_crops.append(crop)
                overlapping_offsets[b, i, :] = [
                    new_x_offset,
                    new_y_offset,
                    new_z_offset,
                ]
            overlapping_crops.append(np.stack(per_image_crops, axis=0))

        return np.stack(overlapping_crops, axis=0), overlapping_offsets

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        if data is None:
            raise ValueError(f"No data found for key {self.data_key}")

        reference_crops, reference_offsets = self.get_reference_crops(
            data
        )  # [B, N_CROPS_PER_IMAGE, C, X_subcrop, Y_subcrop, Z_subcrop], [B, N_CROPS_PER_IMAGE, 3]

        overlapping_crops, overlapping_offsets = self.get_overlapping_crops(
            data, reference_offsets
        )  # [B, N_CROPS_PER_IMAGE, C, X_subcrop, Y_subcrop, Z_subcrop], [B, N_CROPS_PER_IMAGE, 3]

        assert np.all(
            np.max(np.abs(overlapping_offsets - reference_offsets), axis=(0, 1))
            < self.crop_size
        ), "Overlapping offsets are in the expected range!"

        # target_crops, gt_overlap = self.get_target_crops(data)
        # # target_crops: [B, N_target_subcrops, C, X_subcrop, Y_subcrop, Z_subcrop]
        # # gt_overlap: [B, N_target_subcrops, N_base_subcrops]
        B = reference_crops.shape[0]
        n_crops_per_image = reference_crops.shape[1]

        reference_crop_index = (
            B * n_crops_per_image
        )  # Should be B * N_CROPS_PER_IMAGE anyway

        # Flatten for potential transform and later concat
        reference_crops = rearrange(reference_crops, "b n c x y z  -> (b n) c x y z")
        overlapping_crops = rearrange(
            overlapping_crops, "b n c x y z  -> (b n) c x y z"
        )

        if self.aug == "train":
            reference_crops = self.crop_augmentations(
                **{"data": reference_crops, "seg": None}
            )["data"]
            overlapping_crops = self.crop_augmentations(
                **{"data": overlapping_crops, "seg": None}
            )["data"]

        joint_crops_flat = np.concatenate([reference_crops, overlapping_crops], axis=0)

        batch = {
            "all_crops": joint_crops_flat,
            "reference_crop_index": reference_crop_index,
            "batch_size": B,
            "n_crops_per_image": n_crops_per_image,
        }

        return batch


if __name__ == "__main__":
    test_volume = np.random.rand(2, 1, 192, 192, 64)
    inp_dict = {"data": test_volume}
    trafo = SimCLRTransform(
        crop_size=(64, 64, 64),
        data_key="data",
        aug="train",
        crop_count_per_image=3,
        min_overlap_ratio=0.5,
    )
    res = trafo(**inp_dict)
    print(res["reference_crop_index"])
