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
from copy import deepcopy
from dataclasses import asdict
from functools import partial
import multiprocessing
from pathlib import Path
import shutil
from typing import Union

import blosc2

import nnssl
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *


from nnssl.data.raw_dataset import Dataset, IndependentImage
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.paths import nnssl_preprocessed, nnssl_raw
from nnssl.preprocessing.cropping.cropping import crop_to_nonzero
from nnssl.preprocessing.normalization.normalization_schemes import apply_normalization
from nnssl.preprocessing.resampling.default_resampling import compute_new_shape, get_resampling_scheme
from nnssl.training.dataloading.dataset import nnSSLDatasetBlosc2
from nnssl.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnssl.data.utils import get_train_dataset


def normalize(
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


def preprocess_case(
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

    # crop, remember to store size before cropping!
    shape_before_cropping = data.shape[1:]
    properties["shape_before_cropping"] = shape_before_cropping
    # this command will generate a segmentation. This is important because of the nonzero mask which we may need
    data, masks, bbox = crop_to_nonzero(data, masks)
    properties["bbox_used_for_cropping"] = bbox
    properties["shape_after_cropping_and_before_resampling"] = data.shape[1:]

    # resample
    target_spacing = config_plan.spacing  # this should already be transposed

    if len(target_spacing) < len(data.shape[1:]):
        # target spacing for 2d has 2 entries but the data and original_spacing have three because everything is 3d
        # in 2d configuration we do not change the spacing between slices
        target_spacing = [original_spacing[0]] + target_spacing
    new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

    # normalize
    # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
    # longer fitting the images perfectly!
    norm_mask = masks[0]
    data = normalize(data, norm_mask, config_plan.normalization_schemes, config_plan.use_mask_for_norm)

    old_shape = data.shape[1:]
    resampling_fn = partial(
        get_resampling_scheme(config_plan.resampling_fn_data), **config_plan.resampling_fn_data_kwargs
    )
    data = resampling_fn(data, new_shape, original_spacing, target_spacing)

    if has_masks:
        resampling_mask_fn = partial(
            get_resampling_scheme(config_plan.resampling_fn_mask), **config_plan.resampling_fn_mask_kwargs
        )
        for cnt, mask in enumerate(masks):
            masks[cnt] = resampling_mask_fn(mask, new_shape, original_spacing, target_spacing)
    if verbose:
        print(
            f"old shape: {old_shape}, new_shape: {new_shape}, old_spacing: {original_spacing}, "
            f"new_spacing: {target_spacing}, fn_data: {config_plan.resampling_fn_data}"
        )
    if not has_masks:
        masks = None
    return data, masks


def preprocess_and_save(
    image: IndependentImage,
    output_directory: str,
    plan: Plan,
    config_plan: ConfigurationPlan,
    verbose: bool = True,
):
    """Reads the images and their properties, preprocesses them and saves them to disk. (in a compressed npz)"""
    output_filename = Path(join(output_directory, image.get_output_path()))
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    try:
        rw = plan.image_reader_writer_class()()
        image_path = image.image_path
        data, data_properties = rw.read_images([image_path])
        if image.associated_masks is not None:
            masks = [rw.read_seg(v)[0] for v in asdict(image.associated_masks).values() if v is not None]
        else:
            masks = None
        data, masks = preprocess_case(data, masks, data_properties, plan, config_plan, verbose)
        # print('dtypes', data.dtype, seg.dtype)
        block_size_data, chunk_size_data = nnSSLDatasetBlosc2.comp_blosc2_params(
            data.shape, tuple(config_plan.patch_size), data.itemsize
        )
        if masks is not None:
            block_size_seg, chunk_size_seg = nnSSLDatasetBlosc2.comp_blosc2_params(
                data.shape, tuple(config_plan.patch_size), data.itemsize
            )
            if image.associated_masks.anatomy_mask is not None:
                anat_mask = masks[0]
            else:
                anat_mask = None
            if image.associated_masks.anonymization_mask is not None:
                anon_mask = masks[-1]
            else:
                anon_mask = None
        else:
            block_size_seg, chunk_size_seg = None, None
            anat_mask, anon_mask = None, None

        nnSSLDatasetBlosc2.save_case(
            data,
            anon_mask,
            anat_mask,
            data_properties,
            str(output_filename),
            chunks=chunk_size_data,
            blocks=block_size_data,
            chunks_seg=chunk_size_seg,
            blocks_seg=block_size_seg,
        )
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False


def default_preprocess(
    dataset_name_or_id: Union[int, str],
    configuration_name: str,
    plans_identifier: str,
    part: int,
    total_parts: int,
    num_processes: int,
    verbose: bool = True,
):
    """
    Main function that is called externally.
    Does the preprocessing of the cases found in the dataset_name.
    This is the nnssl version, where we neglect any labels that may be present and create a new dataset.json
    that does not contain label information.
    """
    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    assert isdir(join(nnssl_raw, dataset_name)), "The requested dataset could not be found in nnssl_raw"

    plans_file = join(nnssl_preprocessed, dataset_name, plans_identifier + ".json")
    assert isfile(plans_file), (
        "Expected plans file (%s) not found. Run corresponding nnUNet_plan_experiment " "first." % plans_file
    )
    plan: Plan = Plan.load_from_file(plans_file)
    config_plan: ConfigurationPlan = plan.configurations[configuration_name]

    if verbose:
        print(f"Preprocessing the following configuration: {configuration_name}")
        print(config_plan)

    dataset_json_file = join(nnssl_raw, dataset_name, "dataset.json")
    # ToDo: If the dataset.json does not exist, we create one with images used from imagesTr
    #   If this imagesTr does not exist -> error
    # If the dataset.json holds "images_info" we use the paths from there and ignore potential imagesTr
    dataset_json = load_json(dataset_json_file)
    if "labels" in dataset_json.keys():
        dataset_json.pop("labels")  # Remove the labels from the dataset.json

    output_directory = join(nnssl_preprocessed, dataset_name, config_plan.data_identifier)

    # if isdir(output_directory):
    #     shutil.rmtree(output_directory)

    maybe_mkdir_p(output_directory)

    dataset: Dataset = get_train_dataset(join(nnssl_raw, dataset_name), dataset_json)
    # identifiers = [os.path.basename(i[:-len(dataset_json['file_ending'])]) for i in seg_fnames]
    # output_filenames_truncated = [join(output_directory, i) for i in identifiers]
    pp_dataset = deepcopy(dataset)
    pp_dataset.update_extension(new_extension=".b2nd")
    save_json(pp_dataset.to_dict(relative_paths=True), join(nnssl_preprocessed, dataset_name, "pretrain_data.json"))
    # multiprocessing magic.
    preprocess_and_save_partial = partial(
        preprocess_and_save,
        output_directory=output_directory,
        plan=plan,
        config_plan=config_plan,
        verbose=verbose,
    )
    all_independent_images: list[IndependentImage] = dataset.to_independent_images()
    # ------------------- Optional new splitting into sub-parts ------------------ #
    if total_parts > 1:
        total_images = len(all_independent_images)
        images_per_part = total_images // total_parts
        all_independent_images = all_independent_images[part * images_per_part : (part + 1) * images_per_part]

    if num_processes > 1:
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            r = p.map(preprocess_and_save_partial, all_independent_images)
    else:
        r = [preprocess_and_save_partial(image=img) for img in all_independent_images]

    return


if __name__ == "__main__":
    print("Not intended to be called here!")
