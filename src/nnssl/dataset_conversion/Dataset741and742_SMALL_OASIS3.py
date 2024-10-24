from pathlib import Path
import os
from random import sample
import shutil
from batchgenerators.utilities.file_and_folder_operations import save_json
from tqdm import tqdm
import SimpleITK as sitk


def main():
    path_to_raw_dataset = "/mnt/cluster-data-all/t006d/big_brain/OASIS3"
    out_dir = "/mnt/cluster-data-all/t006d/nnunetv2/raw_data"
    dataset_name_1 = "Dataset741_Small_OASIS3_T1_only"
    dataset_name_2 = "Dataset742_Small_OASIS3_T1_T2"
    # for dataset_name, ending in [(dataset_name_1, "T1w.nii.gz"), (dataset_name_2, ("T1w.nii.gz", "T2w.nii.gz"))]:
    for dataset_name, ending in [(dataset_name_2, ("T2w.nii.gz", "T1w.nii.gz"))]:
        dataset_dir = Path(out_dir) / dataset_name
        out_train_dir = dataset_dir / "imagesTr"

        content = os.listdir(path_to_raw_dataset)
        if isinstance(ending, str):
            all_images = [f for f in content if f.endswith(ending)]
            all_images = sample(all_images, 1000)
        else:
            n_endings = len(ending)
            samples_per_ending = 1000 // n_endings
            all_images = []
            for e in ending:
                print(f"Choosing {samples_per_ending} images with ending {e}")
                imgs_of_ending = [f for f in content if f.endswith(e)]
                chosen_imgs_of_ending = []
                with tqdm(total=samples_per_ending, desc="Searching 3D images") as pbar:
                    while len(chosen_imgs_of_ending) < samples_per_ending and len(imgs_of_ending) > 0:
                        chosen_img = imgs_of_ending.pop()
                        im = sitk.ReadImage(os.path.join(path_to_raw_dataset, chosen_img))
                        if len(im.GetSize()) == 3:
                            chosen_imgs_of_ending.append(chosen_img)
                            pbar.update(1)
                        else:
                            print(f"Image {chosen_img} has {len(im.GetSize())} dimensions, not 3. Skipping")
                    all_images += chosen_imgs_of_ending
        images = all_images

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
