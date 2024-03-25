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
import os.path
from functools import lru_cache
from typing import Union

from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import re

from nnssl.paths import nnssl_raw


def get_filenames_of_train_images(raw_dataset_folder: str, dataset_json: dict = None):
    """
    Returns a dataset, containing the images.
    """
    if dataset_json is None:
        dataset_json = load_json(join(raw_dataset_folder, "dataset.json"))
    raw_image_folder = join(raw_dataset_folder, "imagesTr")

    if "dataset" in dataset_json.keys():
        dataset = dataset_json["dataset"]
        for k in dataset.keys():
            dataset[k]["images"] = [
                os.path.abspath(join(raw_dataset_folder, i)) if not os.path.isabs(i) else i
                for i in dataset[k]["images"]
            ]
    else:
        len_ext = len(dataset_json["file_ending"])
        identifiers = [f[:-len_ext] for f in os.listdir(raw_image_folder) if f.endswith(dataset_json["file_ending"])]
        images = [join(raw_image_folder, i + dataset_json["file_ending"]) for i in identifiers]
        segs = [None for _ in range(len(images))]
        dataset = {i: {"images": [im], "label": se} for i, im, se in zip(identifiers, images, segs)}
        print("First four dataset keys:", list(dataset.keys())[:4])
    return dataset


if __name__ == "__main__":
    print(get_filenames_of_train_images(join(nnssl_raw, "Dataset002_Heart")))
