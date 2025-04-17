# nnssl dataset format
While nnssl started as a fork from nnU-Net the dataset format of nnU-Net is impractical for pre-training.
Instead of relying on the data in a specific folder structure nnssl uses a `pretrain_data.json`.
This `pretrain_data.json` holds details about:
- Paths to the raw images
- Meta-data about the image (e.g. scanner names, subject IDs, session IDs, etc.)
However how


Datasets consist of three components: raw images, corresponding segmentation maps and a dataset.json file specifying 
some metadata. While this is more effort than just using plain images (and segmentations) as in nnU-Net this allows you
to use these meta-data to influence how your training is happening. I.e. you could choose to sample not randomly but instead
you could sample uniformly across age. This is not easily possible with nnU-Net as you discard this information.

## Creating a `pretrain_data.json` file
The easiest way to create a `pretrain_data.json` file is through a python script that creates a `Collection` dataclass object.
This is located in `src/nnssl/data/raw_dataset.py`. The `Collection` dataclass has a `.to_dict()` method which will yield a valid `pretrain_data.json` file.
The `Collection` dataclass is a simple dataclass that holds a hierarchical structure of the dataset.
It contains the following fields:
- `collection_index`: The index of the collection. This is a unique identifier for the collection.
- `collection_name`: The name of the collection. This is a human-readable name for the collection.
- `datasets`: A dictionary of sub-datasets

Each `Dataset` is another dataclass object which holds the following fields:
- `dataset_index`: The index of the dataset. This is a unique identifier for the dataset.
- `dataset_info`: A dictionary holding meta information about the dataset. This can be anything you want to store.
- `name`: The name of the dataset. This is a human-readable name for the dataset.
- `subjects`: A dictionary of subjects

Each `Subject` is another dataclass object which holds the following fields:
- `subject_id`: The ID of the subject. This is a unique identifier for the subject in a Dataset -- It doesn't have to be unique across all datasets!
- `subject_info`: A dictionary holding meta information about the subject. This can be e.g. Age/Sex or anything else you want to store.
- `sessions`: A dictionary of Sessions

Each `Session` is another dataclass object which holds the following fields:
- `session_id`: The ID of the session. This is a unique identifier for the session.
- `session_info`: A dictionary holding meta information about the session. This can be anything you want to store.
- `images`: A list of images

Each `Image` is another dataclass object which holds the following fields:
- `name`: The name of the image. This is a human-readable name for the image.
- `modality`: The modality of the image. This is a human-readable name for the modality.
- `image_info`: A dictionary holding meta information about the image (e.g. Scanner Name, Acquisition Parameters, infos about motion artifacts). Can be anything you want to store.
- `image_path`: The path to the image. This is the location on disk of the image `/some/path/to/img.nii.gz`.
- `associated_masks`: Associated masks are another dataclass, which hold the `anonymization_mask` and the `anatomy_mask`. These are complementary masks (`.nii.gz` files) which can help in loss calculation or in sampling more efficiently. 

Each `AssociatedMasks` dataclass has the fields:
- `anonymization_mask`: Path on disk to the anonymization mask or None
- `anatomy_mask`: Path on disk to the anatomy mask or None

This folder structure mirrors the BIDS structure and easily allows you to maintain the structure of the data in the `pretrain_data.json` file should one want to leverage it during pre-training.

An Example script of this is located in `src/nnssl/data/dataset_conversion/Dataset001_OpenMind.py` which creates a `pretrain_data.json` of the [OpenMind dataset](https://huggingface.co/datasets/AnonRes/OpenMind) from huggingface. 


## File Formatting
Currently it does not matter how your raw data looks like, however it is important that the data is in a somewhat consistent format. It might be problematic if you try to combine too many varying formats like e.g. `.nrrd` and `.nii.gz` and `.mha`, so to be safe we recommend to use one format for all images and segmentations. 

Currently all trainers in nnssl treat each image as a single entity, independent from all other images that exist, however as previously mentioned this loading logic can be overridden by the user.


# Example dataset conversion scripts
In the `dataset_conversion` folder (see [here](../nnssl/dataset_conversion)) are multiple example scripts for converting datasets to `nnssl` format. We also provide a dedicated entrypoint for &*Dataset001_OpenMind* which you can call via `nnssl_convert_openmind --openmind_root_dir </Path/to/root/dir/OpenMind>`. We recommend this script as a starting point for your own dataset conversion.

## Public Service annoucenment
If you find this information lacking, please open an issue and request some more details!
The repository is still in its infancy and we are happy to provide more information if you need it.
