from pathlib import Path
import os
from random import sample
import shutil
from batchgenerators.utilities.file_and_folder_operations import save_json
from tqdm import tqdm


def main():
    path_to_raw_dataset = "/mnt/cluster-data-all/t006d/big_brain/OASIS3"
    out_dir = "/home/tassilowald/Data/Datasets/nnunetv2/nnUNet_raw"
    dataset_name_1 = "Dataset741_Small_OASIS3_T1_only"
    dataset_name_2 = "Dataset742_Small_OASIS3_T1_T2"
    for dataset_name, ending in [(dataset_name_1, "T1w.nii.gz"), (dataset_name_2, ("T1w.nii.gz", "T2.nii.gz"))]:
        dataset_dir = Path(out_dir) / dataset_name
        out_train_dir = dataset_dir / "imagesTr"

        content = os.listdir(path_to_raw_dataset)
        all_images = [f for f in content if f.endswith(ending)]
        images = sample(all_images, 1000)

        numTrain = len(images)

        dataset_json = {
            "name": dataset_name,
            "description": "Anatomical MRIs of the OASIS3 dataset without labels. The dataset is used for pre-text task pretraining.",
            "channel_names": {"0": "someMRI"},
            "file_ending": ".nii.gz",
            "numTraining": numTrain,
            "release": "0.0",
            "licence": "Proprietary -- do not touch without permission",
        }

        os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(out_train_dir, exist_ok=True)
        for image in tqdm(images, desc="Copying images"):
            shutil.copy(os.path.join(path_to_raw_dataset, image), os.path.join(out_train_dir, image))
        save_json(dataset_json, dataset_dir / "dataset.json")


if __name__ == "__main__":
    main()
