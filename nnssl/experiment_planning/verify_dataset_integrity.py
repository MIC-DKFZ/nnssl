#    Copyright 2021 HIP Applied Computer Vision Lab, Division of Medical Image Computing, German Cancer Research Center
#    (DKFZ), Heidelberg, Germany
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
from itertools import repeat
import multiprocessing
import re
from multiprocessing import Pool
from typing import Type

import numpy as np
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *

from nnssl.imageio.base_reader_writer import BaseReaderWriter
from nnssl.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnssl.paths import nnUNet_raw
from nnssl.utilities.utils import get_filenames_of_train_images


def check_image_cases(
    image_files: List[str], expected_num_channels: int, readerclass: Type[BaseReaderWriter]
) -> bool:
    """
    Checks if the image cases are consistent with each other.
    """

    rw = readerclass()
    ret = True

    images, _ = rw.read_images(image_files)

    if np.any(np.isnan(images)):
        print(
            f"Images contain NaN pixel values. You need to fix that by "
            f"replacing NaN values with something that makes sense for your images!\nImages:\n{image_files}"
        )
        ret = False

    # check modalities
    if not len(images) == expected_num_channels:
        print(
            "Error: Unexpected number of modalities. \nExpected: %d. \nGot: %d. \nImages: %s\n"
            % (expected_num_channels, len(images), image_files)
        )
        ret = False
    return ret


def verify_dataset_without_labels_integrity(folder: str, num_processes: int = 8) -> None:
    """
    folder needs the imagesTr, imagesTs and labelsTr subfolders. There also needs to be a dataset.json
    checks if the expected number of training cases and labels are present
    for each case, if possible, checks whether the pixel grids are aligned
    checks whether the labels really only contain values they should
    :param folder:
    :return:
    """
    assert isfile(join(folder, "dataset.json")), f"There needs to be a dataset.json file in folder, folder={folder}"
    dataset_json = load_json(join(folder, "dataset.json"))
    dataset = get_filenames_of_train_images(folder, dataset_json)
    image_files = [v["images"] for v in dataset.values()]
    reader_writer_class = determine_reader_writer_from_dataset_json(dataset_json, image_files[0])

    num_modalities = len(
        dataset_json["channel_names"].keys()
        if "channel_names" in dataset_json.keys()
        else dataset_json["modality"].keys()
    )

    if not "dataset" in dataset_json.keys():
        assert isdir(join(folder, "imagesTr")), f"There needs to be a imagesTr subfolder in folder, folder={folder}"

    # make sure all required keys are there
    dataset_keys = list(dataset_json.keys())
    required_keys = ["channel_names", "numTraining", "file_ending"]
    assert all([i in dataset_keys for i in required_keys]), (
        "not all required keys are present in dataset.json."
        "\n\nRequired: \n%s\n\nPresent: \n%s\n\nMissing: "
        "\n%s\n\nUnused by nnU-Net:\n%s"
        % (
            str(required_keys),
            str(dataset_keys),
            str([i for i in required_keys if i not in dataset_keys]),
            str([i for i in dataset_keys if i not in required_keys]),
        )
    )

    expected_num_training = dataset_json["numTraining"]

    # check if the right number of training cases is present
    assert (
        len(dataset) == expected_num_training
    ), "Did not find the expected number of training cases " "(%d). Found %d instead.\nExamples: %s" % (
        expected_num_training,
        len(dataset),
        list(dataset.keys())[:5],
    )

    # check whether only the desired labels are present
    if num_processes > 1:
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            # check whether shapes and spacings match between images and labels
            result = p.starmap(
                check_image_cases,
                zip(
                    image_files,
                    repeat(num_modalities),
                    repeat(reader_writer_class),
                ),
            )
            if not all(result):
                raise RuntimeError(
                    "Some images have errors. Please check text output above to see which one(s) and what's going on."
                )
    else:
        for i in range(len(image_files)):
            if not check_image_cases(image_files[i], num_modalities, reader_writer_class):
                raise RuntimeError(
                    "Some images have errors. Please check text output above to see which one(s) and what's going on."
                )

    # check for nans
    # check all same orientation nibabel
    print("\n####################")
    print(
        "verify_dataset_integrity Done. \nIf you didn't see any error messages then your dataset is most likely OK!"
    )
    print("####################\n")


def verify_dataset_integrity(folder: str, num_processes: int = 8) -> None:
    print("Not using any labels (if present), assuming this is a dataset without labels.")
    verify_dataset_without_labels_integrity(folder, num_processes)
    return


if __name__ == "__main__":
    # investigate geometry issues
    example_folder = join(nnUNet_raw, "Dataset250_COMPUTING_it0")
    num_processes = 6
    verify_dataset_integrity(example_folder, num_processes)
