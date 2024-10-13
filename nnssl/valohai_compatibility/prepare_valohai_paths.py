import tarfile
import zipfile
import numpy as np
from valohai.config import is_running_in_valohai
from valohai.paths import get_inputs_path, get_outputs_path
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
import os
from pathlib import Path
import shutil
from tqdm import tqdm
from nnssl.paths import nnssl_raw, nnssl_preprocessed
import SimpleITK as sitk
import datetime
import multiprocessing
from tqdm.contrib.concurrent import process_map
from loguru import logger
import shutil


def file_is_3d(file: str) -> bool:
    """Check if the file is not a 4D file."""
    try:
        im = sitk.ReadImage(file)
        dim = sitk.GetArrayFromImage(im).ndim
    except:
        # Might be non-orthonormal so we want to remove it as well!
        return False
    return dim == 3


def get_broken_pp_identifiers(flat_path: str) -> list[str]:
    """Get all identifiers that are used for preprocessing."""
    npzs = [f for f in os.listdir(flat_path) if f.endswith(".npz")]
    broken_identifiers = []
    for npz in tqdm(npzs, desc="Checking for broken files"):
        try:
            file = np.load(os.path.join(flat_path, npz), "r")
            for k in file.keys():
                _ = file[k]  # This will raise an error if the file is broken

        except zipfile.BadZipFile:
            broken_identifiers.append(npz)
            broken_identifiers.append(npz[:-4] + ".pkl")
    logger.info("Found ", len(broken_identifiers), "broken files.")
    return broken_identifiers


def decompress_file(file_path, target_path):
    try:
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(target_path)
    except Exception as e:
        logger.warning(f"Failed to decompress {file_path} to {target_path}.")
        logger.error(str(e))


def copy_files(file_path, target_path):
    try:
        Path(target_path).parent.mkdir
        shutil.copy(file_path, target_path)
    except shutil.SameFileError:
        logger.warning(f"File {file_path} already exists in {target_path}. Skipping.")


def copy_del_files(file_path, target_path):

    target_dir = Path(target_path).parent
    target_dir.mkdir(exist_ok=True, parents=True)
    if os.path.exists(file_path):
        if os.path.exists(target_dir):
            try:
                shutil.copy(file_path, target_path)
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Failed to remove {file_path}.")
                    logger.error(str(e))
            except Exception as e:
                logger.error("Some error occured:")
                logger.error(str(e))
        else:
            logger.warning(f"Target path {target_path} does not exist.")
    else:
        logger.warning(f"Src file: {file_path} does not exist.")


def move_files(file_path, target_path):
    try:
        Path(target_path).parent.mkdir
        shutil.move(file_path, target_path)
    except shutil.SameFileError:
        logger.warning(f"File {file_path} already exists in {target_path}. Skipping.")


def copy_to_target_and_maybe_decompress_files(path_to_content: str, target_path: str) -> None:
    """Decompress all files in the folder.
    Return if files were compressed. If compressed we skip checking for broken files
    """

    # ----------------------- Decompress files to temp path ---------------------- #
    files_to_extract = [f for f in os.listdir(path_to_content) if f.endswith(".tar.gz")]

    # TQDM Multiprocessing
    logger.info(f"Decompressing {len(files_to_extract)} files.")
    with multiprocessing.Pool(21) as p:
        os.mkdir(target_path)  # Make sure the path exists already.
        p.starmap(
            decompress_file,
            [(os.path.join(path_to_content, f), target_path) for f in files_to_extract],
        )
    # Remove the shitty .json files as they might be messed up from the MP decompression!
    [os.remove(os.path.join(target_path, f)) for f in os.listdir(target_path) if f.endswith(".json")]

    # ------------------ Copy over files that are not compressed ----------------- #
    # There should be a clean copy of the json also passed!
    other_files = list(set([f for f in os.listdir(path_to_content) if not f.endswith(".tar.gz")]))
    file_target_pairs = [(os.path.join(path_to_content, f), target_path) for f in other_files]

    # TQDM Multiprocessing
    logger.info(f"Copying over {len(file_target_pairs)} files.")
    for file, target in tqdm(file_target_pairs, desc="Copying files"):
        copy_files(file, target)
    logger.info("Done moving files.")
    return len(files_to_extract) > 0


def remove_broken_files_in_folder(data_folder: str):
    broken_files = get_broken_pp_identifiers(data_folder)
    for f in broken_files:
        os.remove(os.path.join(data_folder, f))
    return


def measure_allocated_space_in_path(path: str) -> str:
    """Measure the allocated space in the path in GB."""
    _, used, _ = shutil.disk_usage(path)
    return str(f"{used / (2**30):.2f}")


def measure_free_diskspace(path: str) -> int:
    """Measure the free space in the path."""
    _, _, free = shutil.disk_usage(path)
    return str(f"{free / (2**30):.2f}")


def mp_move_files(source_path, target_path, n_processes=21):
    src_target_pairs: list[tuple[str, str]] = []
    for file in set(os.listdir(source_path)):  # Only unique files
        cur_path = os.path.join(source_path, file)
        if not os.path.exists(cur_path):
            logger.warning("File does not exist:", cur_path)
            continue
        pp_file_path = file.split("__")
        new_path = os.path.join(target_path, *pp_file_path)
        src_target_pairs.append((cur_path, new_path))
    logger.debug(f"Moving {len(src_target_pairs)} Files")
    with multiprocessing.Pool(n_processes) as p:
        p.starmap(move_files, src_target_pairs)


def mp_copy_files(source_path, target_path, n_processes=21):
    src_target_pairs: list[tuple[str, str]] = []
    for file in set(os.listdir(source_path)):
        cur_path = os.path.join(source_path, file)
        if not os.path.exists(cur_path):
            logger.warning("File does not exist:", cur_path)
            continue
        pp_file_path = file.split("__")
        new_path = os.path.join(target_path, *pp_file_path)
        src_target_pairs.append((cur_path, new_path))
    logger.debug(f"Moving {len(src_target_pairs)} Files")
    with multiprocessing.Pool(n_processes) as p:
        p.starmap(copy_files, src_target_pairs)


def mp_copy_del_files(source_path, target_path, n_processes=21):
    src_target_pairs: list[tuple[str, str]] = []
    for file in set(os.listdir(source_path)):
        cur_path = os.path.join(source_path, file)
        if not os.path.exists(cur_path):
            logger.warning("File does not exist:", cur_path)
            continue
        pp_file_path = file.split("__")
        new_path = os.path.join(target_path, *pp_file_path)
        src_target_pairs.append((cur_path, new_path))
    logger.debug(f"Moving {len(src_target_pairs)} Files")
    with multiprocessing.Pool(n_processes) as p:
        p.starmap(copy_del_files, src_target_pairs)


def assert_one_ckpt_exists(input_paths: str, expects_checkpoint: bool = True):
    all_ckpt_files = [f for f in os.listdir(input_paths) if f.endswith(".pth")]
    if len(all_ckpt_files) == 0:
        logger.info("No checkpoint file found in the input folder.")
        if expects_checkpoint:
            raise FileNotFoundError("No checkpoint file found in the input folder.")
        return
    assert len(all_ckpt_files) == 1, f"Found more than 1 checkpoint file: {all_ckpt_files} checkpoint files."


def prepare_training_paths_on_valohai(continue_training: bool):
    if is_running_in_valohai():
        logger.info("Preparing paths for preprocessing on Valohai.")
        INPUT_ROOT = get_inputs_path()
        nnunet_pp = os.path.join(INPUT_ROOT, "nnssl_preprocessed")
        nnunet_results = os.path.join(INPUT_ROOT, "nnssl_results")
        # We create this outside of valohai to be able to remove files.
        temp_pp_path = "/some_non_existing_temp_dir"
        Path(nnunet_pp).mkdir(exist_ok=True)
        Path(nnunet_results).mkdir(exist_ok=True)
        os.environ["nnssl_preprocessed"] = nnunet_pp
        os.environ["nnssl_results"] = nnunet_results
        input_paths = os.path.join(INPUT_ROOT, "pp-data")
        assert_one_ckpt_exists(input_paths, continue_training)
        logger.info(f"Size of downloaded files in {input_paths}: {measure_allocated_space_in_path(input_paths)} GB")
        logger.info(f"Copying/decompressing files from {input_paths} to {temp_pp_path}.")
        is_zipped = copy_to_target_and_maybe_decompress_files(input_paths, temp_pp_path)
        if not is_zipped:
            logger.info(f"Removing broken files in {temp_pp_path}.")
            remove_broken_files_in_folder(temp_pp_path)
        logger.info(f"Total space used in {temp_pp_path}: {measure_allocated_space_in_path(temp_pp_path)} GB")

        logger.info(f"Copy files from {temp_pp_path} to {nnunet_pp}.")
        mp_copy_del_files(temp_pp_path, INPUT_ROOT)
        # for file in os.listdir(temp_pp_path):
        #     cur_path = os.path.join(temp_pp_path, file)
        #     pp_file_path = file.split("__")
        #     new_path = os.path.join(INPUT_ROOT, *pp_file_path)
        #     Path(new_path).parent.mkdir(exist_ok=True, parents=True)
        #     shutil.move(cur_path, new_path)
        # logger.info(f"Removing temp dir: {temp_pp_path}")
        # shutil.rmtree(temp_pp_path)

        logger.info(f"Total space used in {INPUT_ROOT}: {measure_allocated_space_in_path(INPUT_ROOT)} GB")
        logger.info(f"Total space free: {measure_free_diskspace(INPUT_ROOT)} GB")

    else:
        logger.info("Not on valohai.")
        # Local paths are fine, no need to change anything.
        pass


def prepare_preprocessing_paths_on_valohai(dataset_id: int):
    if is_running_in_valohai():
        logger.info("Preparing paths for preprocessing on Valohai.")
        INPUT_ROOT = get_inputs_path()
        nnssl_raw = os.path.join(INPUT_ROOT, "nnssl_raw")
        nnssl_pp = os.path.join(INPUT_ROOT, "nnssl_preprocessed")
        nnssl_results = os.path.join(INPUT_ROOT, "nnssl_results")
        Path(nnssl_raw).mkdir(exist_ok=True)  # create the folder
        Path(nnssl_pp).mkdir(exist_ok=True)
        Path(nnssl_results).mkdir(exist_ok=True)

        flat_inputs = os.path.join(INPUT_ROOT, "raw-data")
        dataset_json_filepath = os.path.join(flat_inputs, "dataset.json")
        if os.path.exists(dataset_json_filepath):
            dataset_json = load_json(dataset_json_filepath)
        else:
            dataset_json = {
                "channel_names": {"0": "someMRI"},
                "description": "Unlabeled set of datapoints that are used for pre-text task pretraining",
                "file_ending": ".nii.gz",
                "licence": "Proprietary -- do not touch without permission",
                "name": "Some Images",
                "numTraining": 0,
                "release": "0.0",
            }
        logger.info(f"Looking for files ending on {dataset_json['file_ending']} in {flat_inputs}.")
        logger.info(f"Found {len(os.listdir(flat_inputs))}")

        dataset_name = f"Dataset{int(dataset_id):03d}_XYZ".format(dataset_id)

        logger.info("Dataset name:", dataset_name)
        nnunet_raw_dataset = os.path.join(nnssl_raw, dataset_name)
        logger.info(f"Creating folder {nnunet_raw_dataset}.")
        Path(nnunet_raw_dataset).mkdir(exist_ok=True)
        nnunet_raw_dataset_imgs = os.path.join(nnssl_raw, dataset_name, "imagesTr")
        Path(nnunet_raw_dataset_imgs).mkdir(exist_ok=True)

        files = [f for f in os.listdir(flat_inputs) if f.endswith(dataset_json["file_ending"])]
        logger.info(f"Found {len(files)} files ... Copying them to {nnunet_raw_dataset_imgs}.")
        # Move raw-data files over.
        not_3d_files = []
        for f in files:
            if not file_is_3d(os.path.join(flat_inputs, f)):
                not_3d_files.append(f)
                continue
            if "/" in f:
                f_target = f.split("/")[-1]
            else:
                f_target = f
            shutil.copy(os.path.join(flat_inputs, f), os.path.join(nnunet_raw_dataset_imgs, f_target))
        logger.info("Found", len(not_3d_files), "files that are not 3D. Ignoring them.")
        logger.info(f"Moved {len(os.listdir(nnunet_raw_dataset_imgs))} files to {nnunet_raw_dataset_imgs}")
        dataset_json["numTraining"] = len(os.listdir(nnunet_raw_dataset_imgs))
        # Adapt number of training cases accordingly.
        save_json(dataset_json, os.path.join(nnunet_raw_dataset, "dataset.json"))

    else:
        logger.info("Not on valohai.")
        # Local paths are fine, no need to change anything.
        pass


def serialize_files_and_move_to_valohai_outputs(some_file_path: str, meta_data_dict: dict | None = None) -> str:
    """
    Takes a file, removes the data structure and saves it encoded into the output folder.
    This can then be easily reverted in next step and when loading.

    meta_data_dict can be used to e.g. create new datasets and append meta data to the output files
    """
    len_path = len(get_inputs_path().split("/"))
    all_parents = some_file_path.split("/")  # First two will be the /valohai/inputs
    out_filename = "__".join(all_parents[len_path:])
    out_filepath = os.path.join(get_outputs_path(), out_filename)
    shutil.copy(some_file_path, out_filepath)
    if meta_data_dict is not None:
        save_json(meta_data_dict, out_filepath + ".metadata.json")
    return out_filepath


def get_all_file_in_dir(dir_path: str) -> list[str]:
    """Get all path files to the files in the directory and subdirectories."""
    files = []
    for f in Path(dir_path).iterdir():
        if f.is_file():
            if (f.name not in [".DS_Store", "._.DS_Store"]) and (not f.name.endswith(".png")):
                files.append(str(f))
            else:
                continue
        else:
            files += get_all_file_in_dir(f)
    return files


def save_plans_on_valohai(
    path_to_copy: str,
    meta_data_dict: dict | None = None,
):
    if is_running_in_valohai():
        pp_path = path_to_copy
        all_files = get_all_file_in_dir(pp_path)
        all_files = [f for f in all_files if f.endswith(".json")]
        logger.info(f"Found {len(all_files)} plans files.")

        # Save the plans files
        for f in tqdm(all_files):
            out_file_path = serialize_files_and_move_to_valohai_outputs(f)
            save_json(meta_data_dict, out_file_path + ".metadata.json")


def save_files_on_valohai(
    path_to_copy: str, meta_data_dict: dict | None = None, compress_output: bool = False, identifier_tag: str = None
):
    """
    Takes all files that were written into the nnssl_preprocessed folder
    and serializes them to the valohai output folder (if running in valohai).
    """
    if is_running_in_valohai():
        pp_path = path_to_copy
        all_files = get_all_file_in_dir(pp_path)
        if not compress_output:
            for f in tqdm(all_files):
                serialize_files_and_move_to_valohai_outputs(f, meta_data_dict)
        else:
            for f in tqdm(all_files):
                serialize_files_and_move_to_valohai_outputs(f)
            path_containing_outputs = get_outputs_path()
            samples_to_compress = os.listdir(path_containing_outputs)
            n_samples_in_path = len(samples_to_compress)
            timestamp = datetime.datetime.now().strftime("%d_%H_%M_%S")
            if identifier_tag is None:
                filename = f"nnssl_pp_{n_samples_in_path}_{timestamp}"
            else:
                filename = f"nnssl_pp_{identifier_tag}_{n_samples_in_path}_{timestamp}"
            compress_format = "gztar"
            logger.info(f"Compressing {n_samples_in_path} samples to {filename}.{compress_format}")
            shutil.make_archive(
                base_name=os.path.join(path_containing_outputs, filename),
                format=compress_format,
                root_dir=path_containing_outputs,
                base_dir=None,
            )
            logger.info(f"Removing {n_samples_in_path} samples from {path_containing_outputs}.")
            [os.remove(os.path.join(path_containing_outputs, f)) for f in samples_to_compress]
            save_json(
                meta_data_dict,
                os.path.join(path_containing_outputs, filename + f".{compress_format}" + ".metadata.json"),
            )

    else:
        # Do nothing
        return


if __name__ == "__main__":
    example_data_path = Path("/home/tassilowald/Data/pseudo_valohai/examplary_data_to_copy_into_inputs-raw-data")
    all_files_to_copy = list(example_data_path.iterdir())
    for f in all_files_to_copy:
        shutil.copy(f, os.path.join("/home/tassilowald/Data/pseudo_valohai/inputs/raw-data", f.name))

    os.environ["VH_JOB_ID"] = "1"  # Make it look like we are on valohai
    os.environ["VH_INPUTS_DIR"] = "/home/tassilowald/Data/pseudo_valohai/inputs"
    os.environ["VH_OUTPUTS_DIR"] = "/home/tassilowald/Data/pseudo_valohai/outputs"
    os.environ["nnssl_raw"] = "/home/tassilowald/Data/pseudo_valohai/pseudo_raw"
    os.environ["nnssl_preprocessed"] = "/home/tassilowald/Data/pseudo_valohai/pseudo_pp"
    os.environ["nnssl_results"] = "/home/tassilowald/Data/pseudo_valohai/pseudo_res"
    prepare_preprocessing_paths_on_valohai(1)
    save_files_on_valohai(os.environ["nnssl_raw"], {"some": "meta_data"})
    print(nnssl_raw)  # Make sure this is actually overwritten!
    print(nnssl_preprocessed)
    logger.info("Done")
