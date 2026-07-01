#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *


from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.preprocessing.cropping.cropping import crop_to_nonzero
from nnssl.preprocessing.preprocessors.normalize import normalize_arr


def no_resample_preprocess_case(
    data: np.ndarray,
    masks: list[np.ndarray] | None,
    properties: dict,
    plan: "Plan",
    config_plan: "ConfigurationPlan",
    verbose: bool,
):
    # let's not mess up the inputs!
    data = np.copy(data)
    if masks is not None:
        for mask in masks:
            assert (
                data.shape[1:] == mask.shape[1:]
            ), "Shape mismatch between image and associated masks. Please fix your dataset and make use of the --verify_dataset_integrity flag to ensure everything is correct"
        masks = [np.copy(mask) for mask in masks]

    has_masks = masks is not None

    # apply transpose_forward, this also needs to be applied to the spacing!
    data = data.transpose([0, *[i + 1 for i in plan.transpose_forward]])
    if has_masks:
        for cnt, mask in enumerate(masks):
            masks[cnt] = mask.transpose([0, *[i + 1 for i in plan.transpose_forward]])
    original_spacing = [properties["spacing"][i] for i in plan.transpose_forward]
    target_spacing = original_spacing

    # crop, remember to store size before cropping!
    shape_before_cropping = data.shape[1:]
    properties["shape_before_cropping"] = shape_before_cropping
    # this command will generate a segmentation. This is important because of the nonzero mask which we may need
    data, masks, bbox = crop_to_nonzero(data, masks)
    properties["bbox_used_for_cropping"] = bbox
    properties["shape_after_cropping_and_before_resampling"] = data.shape[1:]

    norm_mask = masks[0]
    data = normalize_arr(data, norm_mask, config_plan.normalization_schemes, config_plan.use_mask_for_norm)

    new_shape = data.shape[1:]  # compute_new_shape(data.shape[1:], original_spacing, original_spacing)
    old_shape = data.shape[1:]

    if verbose:
        print(
            f"old shape: {old_shape}, new_shape: {new_shape}, old_spacing: {original_spacing}, "
            f"new_spacing: {target_spacing}, fn_data: {config_plan.resampling_fn_data}"
        )
    if not has_masks:
        masks = None
    return data, masks


if __name__ == "__main__":
    print("Not intended to be called here!")
