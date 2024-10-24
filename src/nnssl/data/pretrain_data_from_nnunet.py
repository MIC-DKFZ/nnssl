from ntpath import join
from nnssl.data.raw_dataset import Dataset
import os
from pathlib import Path


from batchgenerators.utilities.file_and_folder_operations import *

import re

from nnssl.data.raw_dataset import Dataset, Subject, Session, Image, associated_masks, AssociatedMasks


def create_pretrain_json_of_nnunet_dataset(dataset_folder: str) -> dict:
    """
    Creates a pretrain json file.
    """

    dataset_json_path: str = join(dataset_folder, "dataset.json")
    assert os.path.exists(dataset_json_path), f"dataset.json does not exist in {dataset_folder}"
    dataset_json: dict = load_json(dataset_json_path)

    modalities: dict = dataset_json["channel_names"]
    ending: str = dataset_json["file_ending"]

    assert os.path.exists(join(dataset_folder, "imagesTr")), "imagesTr folder does not exist"
    images_folder: str = join(dataset_folder, "imagesTr")
    anon_masks: str = join(dataset_folder, "anon_masksTr")
    anat_masks: str = join(dataset_folder, "anat_masksTr")

    anon_masks_exist = os.path.exists(anon_masks)
    anat_masks_exist = os.path.exists(anat_masks)

    dataset_name = os.path.basename(dataset_folder)
    # nnunet format only if we have a 4 digit number at the end of the image name
    case_ids = list(set(["_".join(f.split("_")[:-1]) for f in os.listdir(images_folder) if f.endswith(ending)]))
    root_path = Path(os.path.join(dataset_name, "imagesTr"))
    anon_path = Path(os.path.join(dataset_name, "anon_masksTr"))
    anat_path = Path(os.path.join(dataset_name, "anat_masksTr"))

    all_subjects: list = []
    for case in case_ids:

        cur_session = Session(session_id=0, session_info=None, images=[])
        cur_subject = Subject(subject_id=case, sessions=cur_session, subject_info=None)

        for mod_id, mod_name in modalities.items():
            cur_img = Image(
                name=f"{case.replace(ending, "")}_{mod_id:04d}",
                image_path=str(root_path / f"{case}_{mod_id:04d}{ending}"),
                modality=mod_name,
                image_info=None,
            )
            associated_mask = AssociatedMasks()
            if anon_masks_exist:
                anon_mask_name = join(anon_path, cur_img.name)
                associated_mask.anonymization_mask = anon_mask_name
            if anat_masks_exist:
                anat_mask_name = join(anat_path, cur_img.name)
                associated_mask.anatomy_mask = anat_mask_name
            cur_img.associated_masks = associated_mask
            cur_session.images.append(cur_img)
            assert os.path.exists(join(images_folder, cur_img.name)), f"Image {cur_img.name} does not exist"
        all_subjects.append(cur_subject)
    dataset = Dataset(name=dataset_name, id=int(dataset_name.split("_")[0][-3:]), subjects=all_subjects)
    save_json(dataset.to_dict(), join(dataset_folder, "pretrain_data.json"))
    return dataset


def create_pretrain_json_of_flat_dataset(dataset_json_path: str, flat_data_path: str) -> dict:

    dataset_json_path: str = join(dataset_json_path)
    assert os.path.exists(dataset_json_path), f"dataset.json does not exist: {dataset_json_path}"
    dataset_json: dict = load_json(dataset_json_path)

    modalities: dict = dataset_json["channel_names"]
    ending: str = dataset_json["file_ending"]
    dataset_folder = os.path.dirname(flat_data_path)

    images_folder: str = flat_data_path
    dataset_name = dataset_json["name"]

    all_subjects: list = []
    for case in os.listdir(flat_data_path):
        case_name = case.replace(ending, "")

        cur_session = Session(session_id=0, session_info=None, images=[])
        cur_subject = Subject(subject_id=case, sessions=cur_session, subject_info=None)

        for mod_id, mod_name in modalities.items():
            cur_img = Image(
                name=case_name,
                image_path=join(flat_data_path, case),
                modality=list(modalities.values())[0],
                image_info=None,
            )
            associated_mask = AssociatedMasks()
            cur_img.associated_masks = associated_mask
            cur_session.images.append(cur_img)
            assert os.path.exists(join(images_folder, cur_img.name)), f"Image {cur_img.name} does not exist"
        all_subjects.append(cur_subject)
    dataset = Dataset(name=dataset_name, id=int(dataset_name.split("_")[0][-3:]), subjects=all_subjects)
    save_json(dataset.to_dict(), join(dataset_folder, "pretrain_data.json"))
    return dataset


if __name__ == "__main__":
    pass
