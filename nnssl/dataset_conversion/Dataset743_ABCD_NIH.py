import os
import tarfile
from pathlib import Path
from tqdm import tqdm
import shutil
from multiprocessing import Pool

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


def move_to_cluster():
    target_path = path_to_abcd / Path("abcd_bids")
    cluster_target_path = Path("/mnt/cluster-data-all/t006d/nnunetv2/nnssl_raw")
    cluster_target_path.mkdir(exist_ok=True, parents=True)
    all_nifti_paths = get_nii_files_in_subtree(target_path)
    source_target_pairs = [
        (str(source_path), str(cluster_target_path / (Path(source_path).name))) for source_path in all_nifti_paths
    ]
    print("Moving to cluster")
    with Pool(24) as pool:
        pool.starmap(copy_over_files, source_target_pairs)
    # for src_path, tgt_path in tqdm(source_target_pairs):
    #     copy_over_files(src_path, tgt_path)


if __name__ == "__main__":
    # main()
    move_to_cluster()
