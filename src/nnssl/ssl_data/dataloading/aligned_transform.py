from typing import Literal, Tuple, List, Union

import numpy as np
from batchgenerators.transforms.utility_transforms import NumpyToTensor

from loguru import logger
from batchgenerators.transforms.noise_transforms import (
    GaussianNoiseTransform,
    GaussianBlurTransform,
    RicianNoiseTransform,
)
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform,
    ContrastAugmentationTransform,
    GammaTransform,
)
from batchgenerators.transforms.spatial_transforms import (
    MirrorTransform,
    SpatialTransform,
)
from batchgenerators.transforms.resample_transforms import (
    SimulateLowResolutionTransform,
)
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose

from nnssl.ssl_data.dataloading.overlap_util import make_overlapping_crops_w_bbox


class OneOf(AbstractTransform):
    """
    Applies one of the given transforms with equal probability.
    """

    def __init__(self, transforms: List[AbstractTransform]):
        super().__init__()
        self.transforms = transforms

    def __call__(self, **data_dict):
        transform = self.transforms[np.random.choice(len(self.transforms))]
        return transform(**data_dict)


class OverlapTransform(AbstractTransform):
    def __init__(
        self,
        train: Literal["train", "none"],
        data_key: str,
        initial_patch_size: Tuple[int, ...],
        patch_size: Union[np.ndarray, Tuple[int, ...]],
        rotation_for_DA: dict = None,
        mirror_axes: Tuple[int, ...] = (0, 1, 2),
        do_dummy_2d_data_aug: bool = False,  # 2d data augmentation is not supported yet
        order_resampling_data: int = 3,
        order_resampling_seg: int = 1,
        border_val_seg: int = -1,
        use_mask_for_norm: Union[List[bool], bool] = False,
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
        self.crop_size = patch_size
        self.initial_patch_size = initial_patch_size

        self.min_overlap_ratio = 0.4
        self.max_overlap_ratio = 0.8

        self.aug: str = train
        # Normally we would like to know which axes have the same spacing -- Otherwise rotating the crop will not work.
        #   However since our current plan involves leaving spacing ISO we just add this warning to not forget it.
        logger.warning(
            "Assuming that the axes have the same spacing. If not, the crop rotations will introduce changing spacings."
        )

        if self.aug == "train":
            self.spatial_augmentations: Compose = Compose(
                [
                    MirrorTransform(axes=mirror_axes),
                    SpatialTransform(
                        self.initial_patch_size,
                        patch_center_dist_from_border=None,
                        do_elastic_deform=False,
                        do_rotation=True,
                        p_rot_per_axis=1,
                        do_scale=True,
                        scale=(0.7, 1.4),
                        border_mode_data="constant",
                        border_cval_data=0,
                        order_data=order_resampling_data,
                        border_mode_seg="constant",
                        border_cval_seg=border_val_seg,
                        order_seg=order_resampling_seg,
                        random_crop=False,  # random cropping is part of our dataloaders
                        p_el_per_sample=0,
                        p_scale_per_sample=0.2,
                        p_rot_per_sample=0.2,
                        independent_scale_for_each_axis=False,
                    ),
                    SimulateLowResolutionTransform(
                        zoom_range=(0.5, 1),
                        per_channel=True,
                        p_per_channel=0.5,
                        order_downsample=0,
                        order_upsample=3,
                        p_per_sample=0.1,
                        ignore_axes=None,
                    ),
                ]
            )

            self.crop_augmentations: Compose = Compose(
                [
                    OneOf(
                        [
                            RicianNoiseTransform(
                                noise_variance=(0, 0.1),
                                p_per_sample=0.1,
                            ),
                            GaussianNoiseTransform(p_per_sample=0.1),
                        ]
                    ),
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
                    GammaTransform(
                        (0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1
                    ),
                    GammaTransform(
                        (0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3
                    ),
                ]
            )
        else:
            self.spatial_augmentations: Compose = Compose(
                [
                    SpatialTransform(
                        self.initial_patch_size,
                        do_elastic_deform=False,
                        do_rotation=False,
                        do_scale=False,
                        random_crop=False,
                    )  # do a center crop of patch size 256
                ]
            )

            self.crop_augmentations: Compose = Compose([])  # do nothing if not training

        self.to_tensor = NumpyToTensor(
            keys=["all_crops", "rel_bboxes"], cast_to="float"
        )

    def make_base_crop(self, data_dict):
        """
        Splits the data into reference crops.
        Returns all crops and offsets so overlapping crops can be made.
        Note:
        In this form, multiple crops per image can from close proximity - this might lead to false negatives in the contrastive loss,
        but shouldn't be much of an issue.

        :param data_dict: dict[data_key] = [B, C, X, Y, Z] data to make into a base crop.
        :return: data_dict
        """
        return self.spatial_augmentations(**data_dict)

    def make_overlapping_crops_w_bbox(self, data):
        """
        Make overlapping crops with bounding boxes.
        :param data: [B, C, X, Y, Z] data to make into overlapping crops.
        :return: (crop_1, rel_bbox_1, crop_2, rel_bbox_2)

        - crop_1: [B, C, X_subcrop, Y_subcrop, Z_subcrop]
        - rel_bbox_1: [B, 6] relative bounding box coordinates for crop_1
        - crop_2: [B, C, X_subcrop, Y_subcrop, Z_subcrop]
        - rel_bbox_2: [B, 6] relative bounding box coordinates for crop_2
        The bounding boxes are relative to the original data size and have the format:
            [x_min, y_min, z_min, x_max, y_max, z_max] where the coordinates are normalized
            to [0, 1] range based on the original data size.
        """

        # Make overlapping crops with bounding boxes
        (crop_1, crop_2), (rel_bbox_1, rel_bbox_2) = make_overlapping_crops_w_bbox(
            batch=data,
            patch_size=self.crop_size,
            min_overlap=self.min_overlap_ratio,
            max_overlap=self.max_overlap_ratio,
        )

        crops = np.concatenate([crop_1, crop_2], axis=0)
        rel_bboxes = np.concatenate([rel_bbox_1, rel_bbox_2], axis=0)

        return crops, rel_bboxes

    def __call__(self, **data_dict):
        base_crops = self.make_base_crop(data_dict)

        data = base_crops[self.data_key]

        b = data.shape[0]
        n_crops_per_image = 2

        consistent_crops, rel_bboxes = self.make_overlapping_crops_w_bbox(data)

        if self.aug == "train":
            consistent_crops = self.crop_augmentations(
                **{"data": consistent_crops, "seg": None}
            )["data"]

        batch = {
            "all_crops": consistent_crops,
            "rel_bboxes": rel_bboxes,
            "batch_size": b,
            "n_crops_per_image": n_crops_per_image,
        }

        return self.to_tensor(**batch)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    _b = 2

    # test_volume = np.random.randn(_b, 1, 256, 256, 256)
    test_volume = np.zeros(
        (_b, 1, 256, 256, 256), dtype=np.float32
    )  # Create a zero volume

    # make a square in the middle of the volume for each item
    test_volume[:, 0, :, 20:40, 20:40] = 1  # square in the middle
    test_volume[:, 0, :, 20:40, 60:80] = 2  # another square in the middle
    test_volume[:, 0, :, 20:40, 100:120] = 3  # another square in the middle
    test_volume[:, 0, :, 20:40, 140:160] = 4  # another square in the middle
    test_volume[:, 0, :, 20:40, 180:200] = 5  # another square in the middle
    test_volume[:, 0, :, 20:40, 220:240] = 6  # another square in the middle

    inp_dict = {"data": test_volume}
    trafo = OverlapTransform(
        train="train",
        data_key="data",
        initial_patch_size=(256, 256, 256),
        patch_size=(160, 160, 160),
        rotation_for_DA={"x": (0, 30), "y": (0, 30), "z": (0, 30)},
        mirror_axes=(0, 1, 2),
        do_dummy_2d_data_aug=False,
        order_resampling_data=3,
        order_resampling_seg=1,
        border_val_seg=-1,
        use_mask_for_norm=False,
    )
    res = trafo(**inp_dict)
    print(res["all_crops"].shape)  # Should print shape of the crops
    print(res["rel_bboxes"].shape)  # Should print shape of the bounding boxes

    print(res["rel_bboxes"])  # Print the relative bounding boxes

    # for the item 1, plot the 2 crops and draw a rectangle around the bounding box
    _, axs = plt.subplots(2, _b, figsize=(5 * _b, 5 * 2))
    for _i, _b_idx in enumerate(range(2 * _b)):
        crop = res["all_crops"][_b_idx, 0]  # Assuming single channel
        bbox = res["rel_bboxes"][_b_idx]

        # get the bounding box coordinates in absolute pixel values
        abs_bbox = [
            int(bbox[0] * crop.shape[0]),
            int(bbox[1] * crop.shape[1]),
            int(bbox[2] * crop.shape[2]),
            int(bbox[3] * crop.shape[0]),
            int(bbox[4] * crop.shape[1]),
            int(bbox[5] * crop.shape[2]),
        ]
        # get only the shared area of the crop using the bounding box
        more_crop = crop[
            abs_bbox[0] : abs_bbox[3],
            abs_bbox[1] : abs_bbox[4],
            abs_bbox[2] : abs_bbox[5],
        ]

        axs[_i // 2, _i % 2].imshow(
            more_crop[:, :, more_crop.shape[2] // 2], cmap="gray"
        )
        axs[_i // 2, _i % 2].set_title(f"Crop {_b_idx + 1} - {abs_bbox}")

    plt.tight_layout()
    plt.show()
