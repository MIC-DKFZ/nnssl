from pathlib import Path

import numpy as np
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
    extracted_fomo_task_1_dir = str(Path(".").resolve())
    nnunet_dataset_name = "FOMO60k3"
    nnunet_dataset_id = 603
    dataset_name = f"Dataset{nnunet_dataset_id:03d}_{nnunet_dataset_name}"
    dataset_dir = join(nnUNet_raw, dataset_name)
    maybe_mkdir_p(dataset_dir)

    dataset = {}
    casenames = list(
        map(
            lambda _pth: str(_pth.name),
            Path(extracted_fomo_task_1_dir, "preprocessed").glob("sub_*"),
        )
    )
    for c in casenames:
        dataset[c] = {
            "label": join(extracted_fomo_task_1_dir, "labels", c, "ses_1", "label.nii.gz"),
            "images": [
                join(extracted_fomo_task_1_dir, "preprocessed", c, "ses_1", "t1.nii.gz"),
                join(extracted_fomo_task_1_dir, "preprocessed", c, "ses_1", "t2.nii.gz"),
            ],
        }

    labels = {
        "<10": 0,
        "10-20": 1,
        "20-30": 2,
        "30-40": 3,
        "40-50": 4,
        "50-60": 5,
        "60-70": 6,
        "70-80": 7,
        "80-90": 8,
        "90-100": 9,
        ">100": 10,
    }

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
        # make a dummy segmentation image with the same size as the images with label.txt number mapped to the class
        label_txt = int(open(label_path.replace("label.nii.gz", "label.txt")).read().strip())
        print(f"Label txt for {label_path}: {label_txt}")
        if label_txt < 10:
            label_class = 0
        elif label_txt < 20:
            label_class = 1
        elif label_txt < 30:
            label_class = 2
        elif label_txt < 40:
            label_class = 3
        elif label_txt < 50:
            label_class = 4
        elif label_txt < 60:
            label_class = 5
        elif label_txt < 70:
            label_class = 6
        elif label_txt < 80:
            label_class = 7
        elif label_txt < 90:
            label_class = 8
        elif label_txt < 100:
            label_class = 9
        else:
            label_class = 10
        print(f"Label class for {label_path}: {label_class}")

        # create a dummy segmentation image with the same size as the images with all voxels set to label_class
        reference_image = sitk.ReadImage(images[0])
        dummy_segmentation = sitk.Image(reference_image.GetSize(), sitk.sitkUInt8)
        dummy_segmentation.SetOrigin(reference_image.GetOrigin())
        dummy_segmentation.SetSpacing(reference_image.GetSpacing())
        dummy_segmentation.SetDirection(reference_image.GetDirection())
        dummy_segmentation += label_class
        sitk.WriteImage(dummy_segmentation, label_path)

        np.array(dummy_segmentation, dtype=float)

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
        {0: "t1", 1: "t2"},
        labels,
        num_training_cases=len(dataset),
        file_ending=".nii.gz",
        regions_class_order=None,
        dataset_name=dataset_name,
        reference="https://www.synapse.org/Synapse:syn64895667/wiki/",
        license="see https://www.synapse.org/Synapse:syn64895667/wiki/",
        dataset=dataset,
        description="This dataset does not copy the data into nnunet format and just links to existing data. "
        "The dataset can only be used from one machine because the paths in the dataset.json are hard coded",
    )
