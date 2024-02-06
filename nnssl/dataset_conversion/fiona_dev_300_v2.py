from pathlib import Path
import shutil
import valohai
from batchgenerators.utilities.file_and_folder_operations import save_json


if __name__ == "__main__":
    train_data_path = Path(valohai.inputs("raw-data").path()).parent

    dataset_json = {
        "channel_names": {"0": "someMRI"},
        "file_ending": ".nii.gz",
        "numTraining": len([sdf for sdf in train_data_path.iterdir() if sdf.name.endswith("nii.gz")]),
        "name": "Cases of Floys initial Dataset",
        "release": "0.0",
        "licence": "Proprietary -- do not touch without permission",
        "description": "Unlabeled set of datapoints that are used for pre-text task pretraining",
    }
    # Why do you give me the first item in the dataset instead of the actual directory :'(
    # Why are you like this.
    meta_data_json = {"valohai.dataset-versions": ["dataset://fiona-300-dev/v2"]}
    for sdf in train_data_path.iterdir():
        sdf_name = sdf.name
        if sdf_name.endswith("nii.gz"):
            shutil.copy(sdf, Path("/valohai/outputs") / sdf_name)
            save_json(meta_data_json, Path("/valohai/outputs") / (sdf_name + ".metadata.json"))

    print("Done!")
