import os
from pathlib import Path

from batchgenerators.utilities.file_and_folder_operations import save_json, load_json, join
from tqdm import tqdm

from nnssl.data.raw_dataset import Dataset, Image, Session, Subject, AssociatedMasks, Collection
from nnssl.paths import nnssl_raw

sep = os.path.sep

def _add_pretrain_json():
    scans_lists_dir = Path("/home/j385i/cluster_data/j385i/data/openneuro/scans_lists")
    scans_list_name = "scans_all.json"
    scans_json = load_json(scans_lists_dir / scans_list_name)
    openneuro_dir = Path(os.environ.get("openneuro")) # /mnt/E132-Rohdaten/...
    collection = Collection(
        collection_name="Dataset745_OpenNeuro_v2",
        collection_index=745
    )

    subject_info_keys = ["age", "sex", "handedness", "race", "weight", "bmi", "health_status"]
    image_info_keys = ["derived_from", "is_brain_extract", "num_nonzero_gradients", "manufacturer", "model_name", "phase_encoding_direction", "repetition_time", "echo_time"]

    for dic in tqdm(scans_json):
        relative_scan_path = dic["img_path"]
        modality = dic["modality"]
        subject_info = {k: dic[k] for k in subject_info_keys if k in dic}
        image_info = {k: dic[k] for k in image_info_keys if k in dic}

        parts = relative_scan_path.split(sep)
        dataset_id, subject_id = parts[0], parts[1]
        session_id = "ses-DEFAULT"
        if parts[2].startswith("ses-"):
            session_id = parts[2]
        pre_image_part = parts[-2]
        if "derived_from" in image_info:
            image_name = pre_image_part[:-5] + parts[-1]
        else:
            image_name = parts[-1]

        image_path = str(openneuro_dir / relative_scan_path)

        relative_deface_mask_path = dic.get("deface_mask_path")
        relative_fb_mask_path = dic.get("fb_mask_path")

        anonymization_mask_path = anatomy_mask_path = None
        if relative_deface_mask_path:
            anonymization_mask_path = str(openneuro_dir / relative_deface_mask_path)
        if relative_fb_mask_path:
            anatomy_mask_path = str(openneuro_dir / relative_fb_mask_path)

        associated_masks = AssociatedMasks(anonymization_mask=anonymization_mask_path,
                                           anatomy_mask=anatomy_mask_path)

        if dataset_id not in collection.datasets:
            collection.datasets[dataset_id] = Dataset(dataset_index=dataset_id, name=None, dataset_info=None)

        if subject_id not in collection.datasets[dataset_id].subjects:
            collection.datasets[dataset_id].subjects[subject_id] = Subject(
                subject_id=subject_id,
                subject_info=subject_info
            )
        if session_id not in collection.datasets[dataset_id].subjects[subject_id].sessions:
            collection.datasets[dataset_id].subjects[subject_id].sessions[session_id] = Session(
                session_id=session_id, images=[]
            )
        collection.datasets[dataset_id].subjects[subject_id].sessions[session_id].images.append(
            Image(
                name=image_name,
                image_path=image_path,
                modality=modality,
                image_info=image_info,
                associated_masks=associated_masks
            )
        )
    pretrain_json = collection.to_dict(relative_paths=True)
    save_json(pretrain_json, join(nnssl_raw, "Dataset745_OpenNeuro_v2", "pretrain_data.json"), indent=4, sort_keys=True)


if __name__ == "__main__":
    _add_pretrain_json()
    # _split_pretrain_json(num_chunks=20)
