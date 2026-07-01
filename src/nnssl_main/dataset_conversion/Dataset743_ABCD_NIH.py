import os
import tarfile
from pathlib import Path
from tqdm import tqdm
import shutil
from multiprocessing import Pool
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, isdir, load_json, save_json

from nnssl.data.raw_dataset import Collection, Dataset, Image, Subject, Session

path_to_abcd = Path("/mnt/E132-Rohdaten/wald_collection/ABCD_NIH")


def get_nii_files_in_subtree(root_dir):
    nii_files = []

    # Use scandir to traverse directories recursively
    def scan_directory(directory):
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):  # Recurse into directories
                    scan_directory(entry.path)
                elif entry.is_file() and entry.name.endswith(".nii"):  # Check for .nii files
                    nii_files.append(entry.path)

    scan_directory(root_dir)
    return nii_files


def main():
    path_to_tgz = path_to_abcd / "fmriresults01/abcd-mproc-release5"
    target_path = path_to_abcd / Path("abcd_bids")

    for file_path in tqdm(os.listdir(path_to_tgz)):
        if file_path.endswith(".tgz"):
            with tarfile.open(path_to_tgz / file_path) as tar:
                tar.extractall(target_path)


def copy_over_files(source_path, target_path):
    if not os.path.exists(target_path):
        shutil.copy(source_path, target_path)


def create_collection_json():
    target_path = path_to_abcd / Path("abcd_bids")
    dataset_name = "Dataset743_ABCD_NIH"
    cluster_target_path = Path(f"/mnt/cluster-data-all/t006d/nnunetv2/nnssl_raw/{dataset_name}")
    cluster_target_path.mkdir(exist_ok=True, parents=True)
    all_nifti_paths = get_nii_files_in_subtree(target_path)

    collection: Collection
    collection = Collection(
        collection_index=743,
        collection_name="Dataset743_ABCD_NIH",
    )
    dataset: Dataset
    dataset = Dataset(dataset_index=743, name="ABCD_NIH", dataset_info=None)
    collection.datasets[743] = dataset

    for nifti_path in tqdm(all_nifti_paths, desc="Inserting files into collection."):
        parents = str(nifti_path).split("/")
        subject_id = parents[-4]
        session_id = parents[-3]
        image_name = parents[-1]
        modality = "T1" if "T1w" in image_name else "T2"

        if subject_id not in dataset.subjects:
            subj = Subject(subject_id=subject_id, subject_info=None)
            dataset.subjects[subject_id] = subj
        subj = dataset.subjects[subject_id]
        if session_id not in subj.sessions:
            sess = Session(session_id=session_id)
            subj.sessions[session_id] = sess
        sess = subj.sessions[session_id]

        img = Image(name=image_name, image_path=nifti_path, modality=modality, image_info=None, associated_masks=None)
        sess.images.append(img)
    collection_dict = collection.to_dict(relative_paths=True)
    save_json(collection_dict, os.path.join(cluster_target_path, "pretrain_data.json"))


if __name__ == "__main__":
    # main()
    create_collection_json()
