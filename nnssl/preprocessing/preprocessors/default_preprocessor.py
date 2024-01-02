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
from functools import partial
import multiprocessing
import shutil
from typing import Union

import nnssl
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *


from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.paths import nnssl_preprocessed, nnUNet_raw
from nnssl.preprocessing.cropping.cropping import crop_to_nonzero
from nnssl.preprocessing.normalization.normalization_schemes import apply_normalization
from nnssl.preprocessing.resampling.default_resampling import compute_new_shape, get_resampling_scheme
from nnssl.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnssl.utilities.utils import get_filenames_of_train_images


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
            target_dtype=non_zero_mask[0].dtype,
            use_mask_for_norm=use_mask_for_norm[c],
            non_zero_mask=non_zero_mask[c],
        )
    return data


def preprocess_case(
    data: np.ndarray, properties: dict, plan: "Plan", config_plan: "ConfigurationPlan", verbose: bool
):
    # let's not mess up the inputs!
    data = np.copy(data)

    # apply transpose_forward, this also needs to be applied to the spacing!
    data = data.transpose([0, *[i + 1 for i in plan.transpose_forward]])
    original_spacing = [properties["spacing"][i] for i in plan.transpose_forward]

    # crop, remember to store size before cropping!
    shape_before_cropping = data.shape[1:]
    properties["shape_before_cropping"] = shape_before_cropping
    # this command will generate a segmentation. This is important because of the nonzero mask which we may need
    data, nonzero_mask, bbox = crop_to_nonzero(data, None)
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
    data = normalize(data, nonzero_mask, config_plan.normalization_schemes, config_plan.use_mask_for_norm)

    old_shape = data.shape[1:]
    resampling_fn = partial(
        get_resampling_scheme(config_plan.resampling_fn_data), **config_plan.resampling_fn_data_kwargs
    )
    data = resampling_fn(data, new_shape, original_spacing, target_spacing)
    if verbose:
        print(
            f"old shape: {old_shape}, new_shape: {new_shape}, old_spacing: {original_spacing}, "
            f"new_spacing: {target_spacing}, fn_data: {config_plan.resampling_fn_data}"
        )

    return data


def preprocess_and_save(
    output_filename_truncated: str,
    image_files: List[str],
    plan: Plan,
    config_plan: ConfigurationPlan,
    verbose: bool = True,
):
    """Reads the images and their properties, preprocesses them and saves them to disk. (in a compressed npz)"""
    rw = plan.image_reader_writer_class()()
    data, data_properties = rw.read_images(image_files)
    data = preprocess_case(data, data_properties, plan, config_plan, verbose)
    # print('dtypes', data.dtype, seg.dtype)
    np.savez_compressed(output_filename_truncated + ".npz", data=data)
    write_pickle(data_properties, output_filename_truncated + ".pkl")


def default_preprocess(
    dataset_name_or_id: Union[int, str],
    configuration_name: str,
    plans_identifier: str,
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
    assert isdir(join(nnUNet_raw, dataset_name)), "The requested dataset could not be found in nnUNet_raw"

    plans_file = join(nnssl_preprocessed, dataset_name, plans_identifier + ".json")
    assert isfile(plans_file), (
        "Expected plans file (%s) not found. Run corresponding nnUNet_plan_experiment " "first." % plans_file
    )
    plan: Plan = Plan.load_from_file(plans_file)
    config_plan: ConfigurationPlan = plan.configurations[configuration_name]

    if verbose:
        print(f"Preprocessing the following configuration: {configuration_name}")
        print(config_plan)

    dataset_json_file = join(nnssl_preprocessed, dataset_name, "dataset.json")
    dataset_json = load_json(dataset_json_file)
    dataset_json.pop("labels")  # Remove the labels from the dataset.json

    output_directory = join(nnssl_preprocessed, dataset_name, config_plan.data_identifier)

    if isdir(output_directory):
        shutil.rmtree(output_directory)

    maybe_mkdir_p(output_directory)

    dataset = get_filenames_of_train_images(join(nnUNet_raw, dataset_name), dataset_json)

    # identifiers = [os.path.basename(i[:-len(dataset_json['file_ending'])]) for i in seg_fnames]
    # output_filenames_truncated = [join(output_directory, i) for i in identifiers]

    # multiprocessing magic.
    preprocess_and_save_partial = partial(
        preprocess_and_save,
        plan=plan,
        config_plan=config_plan,
        verbose=verbose,
    )
    all_keys = list(dataset.keys())
    output_filenames = [join(output_directory, i) for i in all_keys]
    all_images = [dataset[i]["images"] for i in all_keys]
    if num_processes > 1:
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            r = p.starmap(preprocess_and_save_partial, zip(output_filenames, all_images))
    else:
        r = [preprocess_and_save_partial(out_fn, imgs) for out_fn, imgs in zip(output_filenames, all_images)]

    return


if __name__ == "__main__":
    print("Not intended to be called here!")
