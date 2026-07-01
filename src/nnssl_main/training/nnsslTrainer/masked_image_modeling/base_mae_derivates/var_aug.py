from typing import List, Literal, Tuple, Union
import torch
from nnssl.experiment_planning.experiment_planners.plan import Plan
from nnssl.ssl_data.data_augmentation.transforms_for_dummy_2d import Convert2DTo3DTransform, Convert3DTo2DTransform
from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import BaseMAETrainer, BaseMAETrainer_BS8_1000ep
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import (
    BrightnessTransform,
    ContrastAugmentationTransform,
    GammaTransform,
)
from batchgenerators.transforms.local_transforms import BrightnessGradientAdditiveTransform, LocalGammaTransform
from batchgenerators.transforms.noise_transforms import (
    MedianFilterTransform,
    GaussianBlurTransform,
    GaussianNoiseTransform,
    BlankRectangleTransform,
    SharpeningTransform,
)
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import (
    SpatialTransform,
    Rot90Transform,
    TransposeAxesTransform,
    MirrorTransform,
)
from batchgenerators.transforms.utility_transforms import (
    OneOfTransform,
    RemoveLabelTransform,
    RenameTransform,
    NumpyToTensor,
)
import numpy as np


aug_level = Literal["off", "low", "medium", "high"]


def _brightnessadditive_localgamma_transform_scale(x, y):
    return np.exp(np.random.uniform(np.log(x[y] // 6), np.log(x[y])))


def _brightness_gradient_additive_max_strength(_x, _y):
    return np.random.uniform(-5, -1) if np.random.uniform() < 0.5 else np.random.uniform(1, 5)


class AugmentedMAETrainer(BaseMAETrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.spatial_aug_level: aug_level = "off"
        self.intensity_aug_level: aug_level = "off"

    def get_training_transforms(
        self,
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: dict,
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        order_resampling_data: int = 3,
        order_resampling_seg: int = 1,
        border_val_seg: int = -1,
        use_mask_for_norm: List[bool] = None,
    ) -> AbstractTransform:
        tr_transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        if self.spatial_aug_level == "low":
            p_scale_per_sample = 0.2
            p_rot_per_sample = 0.2
            scale = (0.7, 1.4)

        elif self.spatial_aug_level == "medium":
            p_scale_per_sample = 0.3
            p_rot_per_sample = 0.3
            scale = (0.6, 1.5)

        elif self.spatial_aug_level == "high":
            p_scale_per_sample = 0.4
            p_rot_per_sample = 0.4
            scale = (0.5, 1.6)

        tr_transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=None,
                do_elastic_deform=False,
                alpha=(0, 0),
                sigma=(0, 0),
                do_rotation=True,
                angle_x=rotation_for_DA["x"],
                angle_y=rotation_for_DA["y"],
                angle_z=rotation_for_DA["z"],
                p_rot_per_axis=0.5,
                do_scale=True,
                scale=scale,
                border_mode_data="constant",
                border_cval_data=0,
                order_data=order_resampling_data,
                # ToDo: Why do we even do scale transforms and do specifically preprocess data? This largely makes no sense, right?
                border_mode_seg="constant",
                border_cval_seg=border_val_seg,
                order_seg=order_resampling_seg,
                random_crop=False,  # random cropping is part of our dataloaders
                p_el_per_sample=0,
                p_scale_per_sample=p_scale_per_sample,
                p_rot_per_sample=p_rot_per_sample,
                independent_scale_for_each_axis=False,  # todo experiment with this
            )
        )

        if do_dummy_2d_data_aug:
            tr_transforms.append(Convert2DTo3DTransform())

        if mirror_axes is not None and len(mirror_axes) > 0:
            tr_transforms.append(MirrorTransform(mirror_axes))

        if self.intensity_aug_level == "low":
            blur_sigma = (0.5, 1.0)
            blur_p_per_sample = 0.2
            median_p_per_sample = 0.2
            brightness_sigma = 0.5
            brightness_p = 0.15

            contrast_range = (0.5, 2)
            contrast_p = 0.06

            gamma_range = (0.7, 1.5)
            gamma_p = 0.1

            brightness_gradient_additive_range = (-0.5, 1.5)  # Only for medium & high
            brightness_gradient_p_per_sample = 0.0  # Only for medium & high

            local_gamma_transform = (-0.25, 1)  # Only for medium & high
            local_gamma_p_per_sample = 0.0  # Only for medium & high

            sharpening_strength = (0.1, 1)  # Only for medium & high
            sharpening_p_per_sample = 0.0  # Only for medium & high
        elif self.intensity_aug_level == "medium":
            # One of the blurs
            blur_sigma = (0.4, 1.1)
            blur_p_per_sample = 0.25
            median_p_per_sample = 0.25
            # Brightness
            brightness_sigma = 0.5
            brightness_p = 0.15

            # Contrast aug
            contrast_range = (0.5, 2)
            contrast_p = 0.13
            # Gamma
            gamma_range = (0.7, 1.5)
            gamma_p = 0.1

            # Brightness gradient
            brightness_gradient_additive_range = (-0.5, 1.5)
            brightness_gradient_p_per_sample = 0.15  # Only for medium & high

            # Local Gamma
            local_gamma_transform = (-0.5, 1.5)  # Only for medium & high
            local_gamma_p_per_sample = 0.15  # Only for medium & high

            sharpening_strength = (0.1, 1)  # Only for medium & high
            sharpening_p_per_sample = 0.1  # Only for medium & high

        elif self.intensity_aug_level == "high":
            # One of the blurs
            blur_sigma = (0.3, 1.2)
            blur_p_per_sample = 0.3
            median_p_per_sample = 0.3
            # Brightness
            brightness_sigma = 0.5
            brightness_p = 0.2

            # Contrast aug
            contrast_range = (0.5, 2)
            contrast_p = 0.06

            # Gamma
            gamma_range = (0.7, 1.5)
            gamma_p = 0.15

            # Brightness gradient
            brightness_gradient_additive_range = (-0.5, 1.5)
            brightness_gradient_p_per_sample = 0.3

            # Local Gamma
            local_gamma_transform = (-0.5, 1.5)  # Only for medium & high
            local_gamma_p_per_sample = 0.3  # Only for medium & high

            # Sharpening
            sharpening_strength = (0.1, 1)  # Only for medium & high
            sharpening_p_per_sample = 0.2  # Only for medium & high

        if self.intensity_aug_level == "off":
            pass
        else:
            tr_transforms.append(
                OneOfTransform(
                    [
                        MedianFilterTransform(
                            (2, 8),
                            same_for_each_channel=False,
                            p_per_sample=median_p_per_sample,
                            p_per_channel=0.5,
                        ),
                        GaussianBlurTransform(
                            blur_sigma,
                            different_sigma_per_channel=True,
                            p_per_sample=blur_p_per_sample,
                            p_per_channel=0.5,
                        ),
                    ]
                )
            )
            tr_transforms.append(
                BrightnessTransform(
                    0,
                    brightness_sigma,
                    per_channel=True,
                    p_per_sample=brightness_p,
                    p_per_channel=0.5,
                ),
            )
            #  Contrast
            tr_transforms.append(
                OneOfTransform(
                    [
                        ContrastAugmentationTransform(
                            contrast_range=contrast_range,
                            preserve_range=True,
                            per_channel=True,
                            data_key="data",
                            p_per_sample=contrast_p,
                            p_per_channel=0.5,
                        ),
                        ContrastAugmentationTransform(
                            contrast_range=contrast_range,
                            preserve_range=False,
                            per_channel=True,
                            data_key="data",
                            p_per_sample=contrast_p,
                            p_per_channel=0.5,
                        ),
                    ]
                )
            )
            # Gamma
            tr_transforms.append(
                GammaTransform(
                    gamma_range,
                    invert_image=True,
                    per_channel=True,
                    retain_stats=True,
                    p_per_sample=gamma_p,
                ),
            )
            # Brightness gradient
            if brightness_gradient_p_per_sample != 0:
                tr_transforms.append(
                    BrightnessGradientAdditiveTransform(
                        _brightnessadditive_localgamma_transform_scale,
                        brightness_gradient_additive_range,
                        max_strength=_brightness_gradient_additive_max_strength,
                        mean_centered=False,
                        same_for_all_channels=False,
                        p_per_sample=brightness_gradient_p_per_sample,
                        p_per_channel=0.5,
                    )
                )
            # Currently causes instability
            # # Local Gamma
            # if local_gamma_p_per_sample != 0:
            #     tr_transforms.append(
            #         LocalGammaTransform(
            #             local_gamma_transform,
            #             gamma=(0.25, ),
            #             p_per_sample=1,  # local_gamma_p_per_sample,
            #             p_per_channel=0.5,
            #         )
            #     )
            # Sharpening
            if sharpening_p_per_sample != 0:
                tr_transforms.append(
                    SharpeningTransform(
                        sharpening_strength,
                        p_per_sample=1,  # sharpening_p_per_sample,
                        p_per_channel=0.5,
                    )
                )

        tr_transforms.append(NumpyToTensor(["data"], "float"))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms


class AugMAETrainer_BS8_ep1000_aug_slow_ioff(AugmentedMAETrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 8
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.spatial_aug_level = "low"
        self.intensity_aug_level = "off"


class AugMAETrainer_BS1_aug_shig_ihig(AugmentedMAETrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 1
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.spatial_aug_level = "high"
        self.intensity_aug_level = "high"


class AugMAETrainer_BS8_ep1000_aug_slow_ilow(AugMAETrainer_BS8_ep1000_aug_slow_ioff):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.spatial_aug_level = "low"
        self.intensity_aug_level = "low"


class AugMAETrainer_BS8_ep1000_aug_slow_imed(AugMAETrainer_BS8_ep1000_aug_slow_ioff):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.spatial_aug_level = "low"
        self.intensity_aug_level = "medium"


class AugMAETrainer_BS8_ep1000_aug_slow_ihig(AugMAETrainer_BS8_ep1000_aug_slow_ioff):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.spatial_aug_level = "low"
        self.intensity_aug_level = "high"


class AugMAETrainer_BS8_ep1000_aug_smed_ioff(AugMAETrainer_BS8_ep1000_aug_slow_ioff):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.spatial_aug_level = "medium"
        self.intensity_aug_level = "off"


class AugMAETrainer_BS8_ep1000_aug_smed_ilow(AugMAETrainer_BS8_ep1000_aug_slow_ioff):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.spatial_aug_level = "medium"
        self.intensity_aug_level = "low"


class AugMAETrainer_BS8_ep1000_aug_smed_imed(AugMAETrainer_BS8_ep1000_aug_slow_ioff):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.spatial_aug_level = "medium"
        self.intensity_aug_level = "medium"


class AugMAETrainer_BS8_ep1000_aug_smed_ihig(AugMAETrainer_BS8_ep1000_aug_slow_ioff):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.spatial_aug_level = "medium"
        self.intensity_aug_level = "high"


class AugMAETrainer_BS8_ep1000_aug_shig_ioff(AugMAETrainer_BS8_ep1000_aug_slow_ioff):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.spatial_aug_level = "high"
        self.intensity_aug_level = "off"


class AugMAETrainer_BS8_ep1000_aug_shig_ilow(AugMAETrainer_BS8_ep1000_aug_slow_ioff):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.spatial_aug_level = "high"
        self.intensity_aug_level = "low"


class AugMAETrainer_BS8_ep1000_aug_shig_imed(AugMAETrainer_BS8_ep1000_aug_slow_ioff):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.spatial_aug_level = "high"
        self.intensity_aug_level = "med"


class AugMAETrainer_BS8_ep1000_aug_shig_ihig(AugMAETrainer_BS8_ep1000_aug_slow_ioff):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.spatial_aug_level = "high"
        self.intensity_aug_level = "high"
