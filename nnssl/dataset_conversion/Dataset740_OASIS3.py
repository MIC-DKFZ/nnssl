from pathlib import Path
import os
from batchgenerators.utilities.file_and_folder_operations import save_json
from tqdm import tqdm


def main():
    path_to_raw_dataset = "/mnt/cluster-data-all/t006d/big_brain/OASIS3"
    out_dir = "/mnt/cluster-data-all/t006d/nnunetv2/raw_data"
    dataset_name = "Dataset740_OASIS3"
    dataset_dir = Path(out_dir) / dataset_name
    out_train_dir = dataset_dir / "imagesTr"

    content = os.listdir(path_to_raw_dataset)
    images = [f for f in content if f.endswith(".nii.gz")]

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
    for image in tqdm(images):
        os.symlink(os.path.join(path_to_raw_dataset, image), os.path.join(out_train_dir, image))
    save_json(dataset_json, dataset_dir / "dataset.json")


if __name__ == "__main__":
    main()
