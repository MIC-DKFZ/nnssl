from dataclasses import asdict
import os
from typing import List

import numpy as np

from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile
from nnssl.ssl_data.dataloading.data_case import DataCase
from nnssl.ssl_data.dataloading.utils import get_case_identifiers


class nnsslDataset(object):
    def __init__(
        self,
        folder: str,
        case_identifiers: List[str] = None,
        num_images_properties_loading_threshold: int = 0,
    ):
        """
        This does not actually load the dataset. It merely creates a dictionary where the keys are training case names and
        the values are dictionaries containing the relevant information for that case.
        dataset[training_case] -> info
        Info has the following key:value pairs:
        - dataset[case_identifier]['properties']['data_file'] -> the full path to the npz file associated with the training case
        - dataset[case_identifier]['properties']['properties_file'] -> the pkl file containing the case properties

        In addition, if the total number of cases is < num_images_properties_loading_threshold we load all the pickle files
        (containing auxiliary information). This is done for small datasets so that we don't spend too much CPU time on
        reading pkl files on the fly during training. However, for large datasets storing all the aux info (which also
        contains locations of foreground voxels in the images) can cause too much RAM utilization. In that
        case is it better to load on the fly.

        If properties are loaded into the RAM, the info dicts each will have an additional entry:
        - dataset[case_identifier]['properties'] -> pkl file content

        IMPORTANT! THIS CLASS ITSELF IS READ-ONLY. YOU CANNOT ADD KEY:VALUE PAIRS WITH nnUNetDataset[key] = value
        USE THIS INSTEAD:
        nnUNetDataset.dataset[key] = value
        (not sure why you'd want to do that though. So don't do it)
        """
        super().__init__()
        # print('loading dataset')
        if case_identifiers is None:
            case_identifiers = get_case_identifiers(folder)
        case_identifiers.sort()

        self.dataset: dict[str, DataCase] = {}
        for c in case_identifiers:
            self.dataset[c] = {}

            data_file = join(folder, f"{c}.npz")
            prop_file = join(folder, f"{c}.pkl")

            self.dataset[c] = DataCase(
                data_file=data_file,
                properties_file=prop_file,
            )

        if len(case_identifiers) <= num_images_properties_loading_threshold:
            for i in self.dataset.keys():
                self.dataset[i].properties = load_pickle(self.dataset[i]["properties_file"])

        self.keep_files_open = ("nnUNet_keep_files_open" in os.environ.keys()) and (
            os.environ["nnUNet_keep_files_open"].lower() in ("true", "1", "t")
        )
        # print(f'nnUNetDataset.keep_files_open: {self.keep_files_open}')

    def __getitem__(self, key):
        ret = {**asdict(self.dataset[key])}
        if "properties" not in ret.keys():
            ret["properties"] = load_pickle(ret["properties_file"])
        return ret

    def __setitem__(self, key, value):
        return self.dataset.__setitem__(key, value)

    def keys(self):
        return self.dataset.keys()

    def __len__(self):
        return self.dataset.__len__()

    def items(self):
        return self.dataset.items()

    def values(self):
        return self.dataset.values()

    def load_case(self, key) -> tuple[np.ndarray, dict]:
        entry = self[key]
        if "open_data_file" in entry.keys():
            data = entry["open_data_file"]
            # print('using open data file')
        elif isfile(entry["data_file"][:-4] + ".npy"):
            data = np.load(entry["data_file"][:-4] + ".npy", "r")
            if self.keep_files_open:
                self.dataset[key]["open_data_file"] = data
                # print('saving open data file')
        else:
            data = np.load(entry["data_file"])["data"]

        return data, entry["properties"]


if __name__ == "__main__":
    # this is a mini test. Todo: We can move this to tests in the future (requires simulated dataset)
    print(0)
