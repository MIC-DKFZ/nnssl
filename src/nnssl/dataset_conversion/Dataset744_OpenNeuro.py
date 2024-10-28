import os
from batchgenerators.utilities.file_and_folder_operations import save_json, load_json, join
from tqdm import tqdm

from nnssl.data.raw_dataset import Dataset, Image, Session, Subject, AssociatedMasks
from nnssl.paths import nnssl_raw

sep = os.path.sep

def _add_pretrain_json():
    scans_json_path = "/home/j385i/cluster_data/j385i/datasets/nnUNet/nnUNet_raw/openneuro_json_dir/scans_ALL.json"
    scans_json = load_json(scans_json_path)
    local_openneuro_dir = os.environ.get("openneuro") # /mnt/E132-Rohdaten/...
    cluster_openneuro_dir = "/omics/groups/OE0441/E132-Rohdaten/WaitingRoom/mic_rocket/raw/openneuro_full_v2" # absolute path from cluster environment
    pretrain_dataset = Dataset(name="Dataset744_OpenNeuro", dataset_index=744)

    subject_info_keys = ["age", "sex", "handedness", "race", "weight", "bmi", "health_status"]
    image_info_keys = ["manufacturer", "model_name", "repetition_time", "echo_time", "slice_thickness"]

    extensions = [".nii", ".nii.gz"]

    for dic in tqdm(scans_json):
        image_path = dic["path"]
        modality = dic["modality"]
        subject_info = {k: dic[k] for k in subject_info_keys}
        image_info = {k: dic[k] for k in image_info_keys}

        parts = image_path.split(sep)
        dataset_id, subject_id = parts[0], parts[1]
        final_subject_id = f"{dataset_id}__{subject_id}"
        session_id = "ses-DEFAULT"
        if parts[2].startswith("ses-"):
            session_id = parts[2]
        image_name = parts[-1]
        image_parent = sep.join(parts[:-1])
        # local_image_path = join(local_openneuro_dir, image_path)
        cluster_image_path = join(cluster_openneuro_dir, image_path)

        anonymization_mask_trunc_name = image_name.split(".nii")[0] + "__DEFACEMASK"
        local_anonymization_mask_trunc_path = join(local_openneuro_dir, image_parent, anonymization_mask_trunc_name)
        # print(local_anonymization_mask_trunc_path)

        anonymization_mask_name = [anonymization_mask_trunc_name + ext for ext in extensions
                                   if os.path.exists(local_anonymization_mask_trunc_path + ext)][0]


        # local_anonymization_mask_path = join(local_openneuro_dir, anonymization_mask_name)
        cluster_anonymization_mask_path = join(cluster_openneuro_dir, image_parent, anonymization_mask_name)

        associated_masks = AssociatedMasks(anonymization_mask=cluster_anonymization_mask_path)

        if final_subject_id not in pretrain_dataset.subjects:
            pretrain_dataset.subjects[final_subject_id] = Subject(
                subject_id=final_subject_id,
                subject_info=subject_info
            )
        if session_id not in pretrain_dataset.subjects[final_subject_id].sessions:
            pretrain_dataset.subjects[final_subject_id].sessions[session_id] = Session(
                session_id=session_id, images=[]
            )
        pretrain_dataset.subjects[final_subject_id].sessions[session_id].images.append(
            Image(
                name=image_name,
                image_path=cluster_image_path,
                modality=modality,
                image_info=image_info,
                associated_masks=associated_masks
            )
        )
    pretrain_json = pretrain_dataset.to_dict()
    save_json(pretrain_json, join(nnssl_raw, "Dataset744_OpenNeuro", "pretrain_data.json"), indent=4, sort_keys=True)


def _split_pretrain_json(num_chunks=20):
    from itertools import batched

    pretrain_json = load_json(join(nnssl_raw, "Dataset744_OpenNeuro", "pretrain_data.json"))
    dataset_name_template = "Dataset{}_OpenNeuro_{}"

    size_per_chunk = len(pretrain_json["subjects"]) // num_chunks + 1

    for i, chunk in enumerate(batched(pretrain_json["subjects"].items(), size_per_chunk)):
        chunk = dict(chunk)
        dataset_name = dataset_name_template.format(744+1+i, i)
        output_dir = join(nnssl_raw, dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = join(output_dir, "pretrain_data.json")

        chunk_pretrain_json = {
            "dataset_index": 744+1+i,
            "dataset_info": None,
            "name": dataset_name,
            "subjects": chunk
        }

        save_json(chunk_pretrain_json, output_file_path, indent=4, sort_keys=True)

        print(f"Saved chunk no.{i} at {output_file_path}. Number of subjects: {len(chunk)}")


if __name__ == "__main__":
    _add_pretrain_json()
    # _split_pretrain_json(num_chunks=20)
