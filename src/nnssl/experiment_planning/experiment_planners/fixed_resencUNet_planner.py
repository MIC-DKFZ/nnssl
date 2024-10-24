from copy import deepcopy
from typing import List, Tuple, Union

import numpy as np
from nnssl.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from nnssl.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props

from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan


class FixedResEncUNetPlanner(ExperimentPlanner):
    def __init__(
        self,
        dataset_name_or_id: Union[str, int],
        gpu_memory_target_in_gb: float = 24,
        preprocessor_name: str = "DefaultPreprocessor",
        plans_name: str = "nnUNetResEncUNetPlans",
        overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
        suppress_transpose: bool = False,
    ):
        super().__init__(
            dataset_name_or_id,
            gpu_memory_target_in_gb,
            preprocessor_name,
            plans_name,
            overwrite_target_spacing,
            suppress_transpose,
        )

        self.UNet_base_num_features = 32
        self.UNet_class = ResidualEncoderUNet
        # the following two numbers are really arbitrary and were set to reproduce default nnU-Net's configurations as
        # much as possible
        self.UNet_reference_val_3d = 680000000
        self.UNet_reference_val_2d = 135000000
        self.UNet_reference_com_nfeatures = 32
        self.UNet_reference_val_corresp_GB = 8
        self.UNet_reference_val_corresp_bs_2d = 12
        self.UNet_reference_val_corresp_bs_3d = 2
        self.UNet_featuremap_min_edge_length = 4
        self.UNet_blocks_per_stage_encoder = (1, 3, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6)
        self.UNet_blocks_per_stage_decoder = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        self.UNet_min_batch_size = 2
        self.UNet_max_features_2d = 512
        self.UNet_max_features_3d = 320

    def get_plans_for_configuration(
        self,
        spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
        median_shape: Union[np.ndarray, Tuple[int, ...], List[int]],
        data_identifier: str,
        approximate_n_voxels_dataset: float,
    ) -> ConfigurationPlan:
        if len(spacing) == 3:
            # For 3D we overwrite whatever else would happen
            spacing = [1.0, 1.0, 1.0]

        assert all([i > 0 for i in spacing]), f"Spacing must be > 0! Spacing: {spacing}"
        # print(spacing, median_shape, approximate_n_voxels_dataset)
        # find an initial patch size
        # we first use the spacing to get an aspect ratio
        tmp = 1 / np.array(spacing)

        # we then upscale it so that it initially is certainly larger than what we need (rescale to have the same
        # volume as a patch of size 256 ** 3)
        # this may need to be adapted when using absurdly large GPU memory targets. Increasing this now would not be
        # ideal because large initial patch sizes increase computation time because more iterations in the while loop
        # further down may be required.
        if len(spacing) == 3:
            initial_patch_size = [160, 160, 160]
        elif len(spacing) == 2:
            initial_patch_size = [round(i) for i in tmp * (2048**2 / np.prod(tmp)) ** (1 / 2)]
        else:
            raise RuntimeError()

        # clip initial patch size to median_shape. It makes little sense to have it be larger than that. Note that
        # this is different from how nnU-Net v1 does it!
        # todo patch size can still get too large because we pad the patch size to a multiple of 2**n
        initial_patch_size = np.array([min(i, j) for i, j in zip(initial_patch_size, median_shape[: len(spacing)])])
        initial_patch_size = [160, 160, 160]
        # use that to get the network topology. Note that this changes the patch_size depending on the number of
        # pooling operations (must be divisible by 2**num_pool in each axis)
        (
            network_num_pool_per_axis,
            pool_op_kernel_sizes,
            conv_kernel_sizes,
            patch_size,
            shape_must_be_divisible_by,
        ) = get_pool_and_conv_props(spacing, initial_patch_size, self.UNet_featuremap_min_edge_length, 999999)
        patch_size = [160, 160, 160]
        # now estimate vram consumption
        num_stages = len(pool_op_kernel_sizes)
        estimate = self.static_estimate_VRAM_usage(
            tuple(patch_size),
            num_stages,
            tuple([tuple(i) for i in pool_op_kernel_sizes]),
            self.UNet_class,
            len(
                self.dataset_json["channel_names"].keys()
                if "channel_names" in self.dataset_json.keys()
                else self.dataset_json["modality"].keys()
            ),
            tuple(
                [
                    min(
                        self.UNet_max_features_2d if len(patch_size) == 2 else self.UNet_max_features_3d,
                        self.UNet_reference_com_nfeatures * 2**i,
                    )
                    for i in range(len(pool_op_kernel_sizes))
                ]
            ),
            self.UNet_blocks_per_stage_encoder[:num_stages],
            self.UNet_blocks_per_stage_decoder[: num_stages - 1],
            len(self.dataset_json["labels"].keys()),
        )

        # how large is the reference for us here (batch size etc)?
        # adapt for our vram target
        reference = (self.UNet_reference_val_2d if len(spacing) == 2 else self.UNet_reference_val_3d) * (
            self.UNet_vram_target_GB / self.UNet_reference_val_corresp_GB
        )

        while estimate > reference:
            # print(patch_size)
            # patch size seems to be too large, so we need to reduce it. Reduce the axis that currently violates the
            # aspect ratio the most (that is the largest relative to median shape)
            axis_to_be_reduced = np.argsort(patch_size / median_shape[: len(spacing)])[-1]

            # we cannot simply reduce that axis by shape_must_be_divisible_by[axis_to_be_reduced] because this
            # may cause us to skip some valid sizes, for example shape_must_be_divisible_by is 64 for a shape of 256.
            # If we subtracted that we would end up with 192, skipping 224 which is also a valid patch size
            # (224 / 2**5 = 7; 7 < 2 * self.UNet_featuremap_min_edge_length(4) so it's valid). So we need to first
            # subtract shape_must_be_divisible_by, then recompute it and then subtract the
            # recomputed shape_must_be_divisible_by. Annoying.
            tmp = deepcopy(patch_size)
            tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
            _, _, _, _, shape_must_be_divisible_by = get_pool_and_conv_props(
                spacing, tmp, self.UNet_featuremap_min_edge_length, 999999
            )
            patch_size[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]

            # now recompute topology
            (
                network_num_pool_per_axis,
                pool_op_kernel_sizes,
                conv_kernel_sizes,
                patch_size,
                shape_must_be_divisible_by,
            ) = get_pool_and_conv_props(spacing, patch_size, self.UNet_featuremap_min_edge_length, 999999)

            num_stages = len(pool_op_kernel_sizes)
            estimate = self.static_estimate_VRAM_usage(
                tuple(patch_size),
                num_stages,
                tuple([tuple(i) for i in pool_op_kernel_sizes]),
                self.UNet_class,
                len(
                    self.dataset_json["channel_names"].keys()
                    if "channel_names" in self.dataset_json.keys()
                    else self.dataset_json["modality"].keys()
                ),
                tuple(
                    [
                        min(
                            self.UNet_max_features_2d if len(patch_size) == 2 else self.UNet_max_features_3d,
                            self.UNet_reference_com_nfeatures * 2**i,
                        )
                        for i in range(len(pool_op_kernel_sizes))
                    ]
                ),
                self.UNet_blocks_per_stage_encoder[:num_stages],
                self.UNet_blocks_per_stage_decoder[: num_stages - 1],
                len(self.dataset_json["labels"].keys()),
            )

        # alright now let's determine the batch size. This will give self.UNet_min_batch_size if the while loop was
        # executed. If not, additional vram headroom is used to increase batch size
        ref_bs = self.UNet_reference_val_corresp_bs_2d if len(spacing) == 2 else self.UNet_reference_val_corresp_bs_3d
        batch_size = round((reference / estimate) * ref_bs)

        # we need to cap the batch size to cover at most 5% of the entire dataset. Overfitting precaution. We cannot
        # go smaller than self.UNet_min_batch_size though
        bs_corresponding_to_5_percent = round(
            approximate_n_voxels_dataset * 0.05 / np.prod(patch_size, dtype=np.float64)
        )
        batch_size = max(min(batch_size, bs_corresponding_to_5_percent), self.UNet_min_batch_size)

        resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs = self.determine_resampling()
        resampling_softmax, resampling_softmax_kwargs = self.determine_segmentation_softmax_export_fn()

        (
            normalization_schemes,
            mask_is_used_for_norm,
        ) = self.determine_normalization_scheme_and_whether_mask_is_used_for_norm()
        num_stages = len(pool_op_kernel_sizes)
        plan = {
            "data_identifier": data_identifier,
            "preprocessor_name": self.preprocessor_name,
            "batch_size": batch_size,
            "patch_size": [160, 160, 160],
            "median_image_size_in_voxels": median_shape,
            "spacing": spacing,
            "normalization_schemes": normalization_schemes,
            "use_mask_for_norm": mask_is_used_for_norm,
            "UNet_class_name": self.UNet_class.__name__,
            "UNet_base_num_features": self.UNet_base_num_features,
            "n_conv_per_stage_encoder": self.UNet_blocks_per_stage_encoder[:num_stages],
            "n_conv_per_stage_decoder": self.UNet_blocks_per_stage_decoder[: num_stages - 1],
            "num_pool_per_axis": network_num_pool_per_axis,
            "pool_op_kernel_sizes": pool_op_kernel_sizes,
            "conv_kernel_sizes": conv_kernel_sizes,
            "unet_max_num_features": self.UNet_max_features_3d if len(spacing) == 3 else self.UNet_max_features_2d,
            "resampling_fn_data": resampling_data.__name__,
            "resampling_fn_data_kwargs": resampling_data_kwargs,
            "batch_dice": False,
        }

        return ConfigurationPlan(**plan)
