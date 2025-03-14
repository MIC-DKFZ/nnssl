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


from batchgenerators.utilities.file_and_folder_operations import load_json, join


from nnssl.data.raw_dataset import Collection
from nnssl.paths import nnssl_raw


def get_pretrain_json_or_create_new(raw_dataset_folder: str) -> dict:
    """Create a pretrain json file if one does not exist given the nnU-Net dataset format."""
    expected_pretrain_json_path = join(raw_dataset_folder, "pretrain_data.json")
    if os.path.exists(expected_pretrain_json_path):
        return load_json(join(raw_dataset_folder, "pretrain_data.json"))
    else:
        raise FileNotFoundError("dataset.json or imagesTr folder does not exist in the given folder")


def get_train_collection(raw_dataset_folder: str) -> Collection:
    """
    Returns a list of all dataset paths, containing paths to the actual files.
    """
    pretrain_dataset = get_pretrain_json_or_create_new(raw_dataset_folder)
    collection = Collection.from_dict(pretrain_dataset)
    return collection


if __name__ == "__main__":
    print(get_train_collection(join(nnssl_raw, "Dataset741_Small_OASIS3_T1_only")))
