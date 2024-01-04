import os
import re
import shutil
from pathlib import Path

from nnssl.paths import nnUNet_raw
from batchgenerators.utilities.file_and_folder_operations import save_json


def convert(src_data_folder: Path, target_data_folder: Path):
    images_tr_dir = target_data_folder / "imagesTr"
    images_tr_dir.mkdir(parents=True, exist_ok=True)
    patient_pattern = r"^[a-z0-9]*$"
    image_pattern = r"^[a-z0-9].*\.nii\.gz$"
    output_pattern = "pat_{:05d}__mri_{:05d}__0000.nii.gz"
    overall_samples = 0
    # -------------------------- Iterate once and count -------------------------- #
    for pat_cnt, pat_id in enumerate(
        [sdf for sdf in src_data_folder.iterdir() if re.match(patient_pattern, sdf.name)]
    ):
        for mri_cnt, mri_id in enumerate([mri for mri in pat_id.iterdir() if re.match(image_pattern, mri.name)]):
            overall_samples += 1
    print(f"Found {overall_samples} samples.")
    # -------------------------- Iterate again and copy -------------------------- #
    pat_json = {}
    for pat_cnt, pat_id in enumerate(
        [sdf for sdf in src_data_folder.iterdir() if re.match(patient_pattern, sdf.name)]
    ):
        pat_json[f"{pat_cnt:05d}"] = {"name": pat_id.name, "images": dict()}
        for mri_cnt, mri_id in enumerate([mri for mri in pat_id.iterdir() if re.match(image_pattern, mri.name)]):
            output_target = images_tr_dir / (output_pattern.format(pat_cnt, mri_cnt))
            pat_json[f"{pat_cnt:05d}"]["images"].update({f"{mri_cnt:05d}": mri_id.name})
            shutil.copy(mri_id, output_target)
    save_json(pat_json, target_data_folder / "patient_id_mapping.json", sort_keys=False)
    dataset_json = {
        "channel_names": {
            0: "someMRI",
        },
        "file_ending": ".nii.gz",
        "num_training_cases": overall_samples,
        "name": "150 cases of Floys initial Dataset",
        "release": "0.0",
        "licence": "Proprietary -- do not touch without permission",
        "description": "Unlabeled set of datapoints that are used for pre-text task pretraining",
    }
    save_json(dataset_json, target_data_folder / "dataset.json", sort_keys=False)


if __name__ == "__main__":
    import os

    train_data_path = Path("/mnt/cluster-checkpoint-all/t006d/rohdaten/mr150")
    out_data_path = Path("/mnt/cluster-data-all/t006d/nnunetv2/raw_data/Dataset737_FloyPrototype")
    assert train_data_path.exists(), f"Train data path {train_data_path} does not exist!"
    out_data_path.mkdir(parents=True, exist_ok=True)
    convert(train_data_path, out_data_path)
    print("Done!")
