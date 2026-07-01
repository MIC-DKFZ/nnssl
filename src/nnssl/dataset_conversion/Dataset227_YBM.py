from pathlib import Path

import SimpleITK as sitk

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    subdirs,
    subfiles,
    maybe_mkdir_p,
)
from nnunetv2.paths import nnUNet_raw


if __name__ == "__main__":
    """
    this dataset does not copy the data into nnunet format and just links to existing data. The dataset can only be
    used from one machine because the paths in the dataset.json are hard coded
    """
    extracted_ybm_dir = "."
    nnunet_dataset_name = "YBM"
    nnunet_dataset_id = 227
    dataset_name = f"Dataset{nnunet_dataset_id:03d}_{nnunet_dataset_name}"
    dataset_dir = join(nnUNet_raw, dataset_name)
    maybe_mkdir_p(dataset_dir)

    dataset = {}
    casenames = list(
        map(
            lambda _pth: str(_pth.name),
            Path(extracted_ybm_dir).glob("BraTS-MET-*"),
        )
    )
    for c in casenames:
        dataset[c] = {
            "label": join(extracted_ybm_dir, c, f"{c}-seg.nii.gz"),
            "images": [
                join(extracted_ybm_dir, c, f"{c}-t1c.nii.gz"),
                join(extracted_ybm_dir, c, f"{c}-t1n.nii.gz"),
                join(extracted_ybm_dir, c, f"{c}-t2f.nii.gz"),
                join(extracted_ybm_dir, c, f"{c}-t2w.nii.gz"),
            ],
        }

    labels = {
        "background": 0,
        "TumourNecrosis": 1,
        "PeritumoralEdema": 2,
        "ContrastEnhancingTumour": 3,
    }

    # resize all dwi to target spacing [1, 1, 1] and then resize all adc to dwi spacing
    # target_spacing = np.array([1, 1, 1])
    for c in casenames:
        label_path = dataset[c]["label"]
        images = dataset[c]["images"]
        sizes = []
        for image_path in images:
            image = sitk.ReadImage(image_path)
            sizes.append(sitk.GetArrayFromImage(image).shape)

            # print the size of the image
            print(f"Size of {image_path}: {sizes[-1]}")

        # print the sizes of all the images and the label
        label_image = sitk.ReadImage(label_path)
        print(
            f"Size of label {label_path}: {sitk.GetArrayFromImage(label_image).shape}"
        )

        sizes.append(sitk.GetArrayFromImage(label_image).shape)

        # find the number of unqiue classes in the label
        unique_classes = set(sitk.GetArrayFromImage(label_image).flatten())
        print(f"Unique classes in label {label_path}: {unique_classes}")

        # check if all sizes are equal
        if not all(size == sizes[0] for size in sizes):
            print("====" * 20)
            print(f"Sizes for {c} are not equal:")
            for i, size in enumerate(sizes):
                print(f"Image {i}: {size}")
            print("====" * 20)

    generate_dataset_json(
        dataset_dir,
        {0: "T1C", 1: "T1N", 2: "T2F", 3: "T2W"},
        labels,
        num_training_cases=len(dataset),
        file_ending=".nii.gz",
        regions_class_order=None,
        dataset_name=dataset_name,
        reference="https://www.nature.com/articles/s41597-024-03021-9",
        license="see https://www.nature.com/articles/s41597-024-03021-9",
        dataset=dataset,
        description="This dataset does not copy the data into nnunet format and just links to existing data. "
        "The dataset can only be used from one machine because the paths in the dataset.json are hard coded",
    )
