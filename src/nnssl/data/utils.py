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

import os
from pathlib import Path
from typing import get_args

from batchgenerators.utilities.file_and_folder_operations import *
from loguru import logger
import numpy as np
import re

from nnssl.data.raw_dataset import Dataset, Subject, Session, Image, associated_masks, AssociatedMasks
from nnssl.paths import nnssl_raw


def get_dataset_from_pretrain_data(pretrain_data: dict, dataset_name: str, dataset_id: int) -> list[str]:
    """
    Read the `pretrain_data.json` and create a proper dataset from it.
    This dataset will allow easy `fingerprinting`, `planning` and `preprocessing`.
    """
    all_subjects = []
    for subject_id, subject_info in pretrain_data.items():
        cur_subject: Subject = Subject(subject_id)
        cur_subject.subject_info = {k: v for k, v in subject_info.items() if k != "sessions"}
        # Sessions are ordered
        all_sessions = []
        for sess_id, sess_imgs in subject_info["sessions"].items():
            cur_sess = Session(sess_id)
            images = []
            for image in sess_imgs["images"]:
                img_path = image["path"]
                modality = image["modality"]
                assoc_masks = image.get("associated_masks", None)
                associated_mask = None
                if assoc_masks is not None:
                    associated_mask = AssociatedMasks()
                    for k in get_args(associated_masks):
                        if k in assoc_masks:
                            associated_mask[k] = assoc_masks[k]
                img = Image(image_path=img_path, modality=modality, associated_masks=associated_mask)
                images.append(img)
            cur_sess.images = images
            all_sessions.append(cur_sess)
        cur_subject.sessions = all_sessions
        all_subjects.append(cur_subject)
    dataset = Dataset(name=dataset_name, id=dataset_id, subjects=all_subjects)
    return dataset




def get_pretrain_json_or_create_new(raw_dataset_folder: str) -> dict:
    """Create a pretrain json file if one does not exist given the nnU-Net dataset format."""
    expected_pretrain_json_path = join(raw_dataset_folder, "pretrain_data.json")
    if os.path.exists(expected_pretrain_json_path):
        return load_json(join(raw_dataset_folder, "pretrain_data.json"))
    elif os.path.exists(join(raw_dataset_folder, "dataset.json")) and os.path.exists(
        join(raw_dataset_folder, "imagesTr")
    ):
        logger.warning(
            f"'pretrain_data.json' does not exist in {raw_dataset_folder}. Creating a new one dervied from 'dataset.json'."
        )
        return create_pretrain_json_of_nnunet_dataset(raw_dataset_folder)
    else:
        raise FileNotFoundError("dataset.json or imagesTr folder does not exist in the given folder")


def get_train_dataset(raw_dataset_folder: str, dataset_json: dict = None) -> Dataset:
    """
    Returns a list of all dataset paths, containing paths to the actual files.
    """
    dataset_name = os.path.basename(raw_dataset_folder)
    dataset_id = int(dataset_name.split("_")[0][-3:])
    pretrain_data = get_pretrain_json_or_create_new(raw_dataset_folder)
    image_paths: str = get_dataset_from_pretrain_data(
        pretrain_data=pretrain_data, dataset_name=dataset_name, dataset_id=dataset_id
    )

    abs_image_paths = [join(os.path.dirname(raw_dataset_folder), ip) for ip in image_paths]

    return abs_image_paths


if __name__ == "__main__":
    print(get_train_dataset(join(nnssl_raw, "Dataset741_Small_OASIS3_T1_only")))
