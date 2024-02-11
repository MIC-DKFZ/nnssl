from valohai.config import is_running_in_valohai
from valohai.paths import get_inputs_path, get_outputs_path
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
import os
from pathlib import Path
import shutil
from tqdm import tqdm
from nnssl.paths import nnUNet_raw, nnssl_preprocessed


def prepare_preprocessing_paths_on_valohai(dataset_id: int):
    if is_running_in_valohai():
        print("Preparing paths for preprocessing on Valohai.")
        INPUT_ROOT = get_inputs_path()
        nnunet_raw = os.path.join(INPUT_ROOT, "nnunet_raw")
        nnunet_pp = os.path.join(INPUT_ROOT, "nnunet_preprocessed")
        nnunet_results = os.path.join(INPUT_ROOT, "nnunet_results")
        Path(nnunet_raw).mkdir(exist_ok=True)  # create the folder
        Path(nnunet_pp).mkdir(exist_ok=True)
        Path(nnunet_results).mkdir(exist_ok=True)
        os.environ["nnUNet_raw"] = nnunet_raw
        os.environ["nnssl_preprocessed"] = nnunet_pp
        os.environ["nnssl_results"] = nnunet_results

        flat_inputs = os.path.join(INPUT_ROOT, "raw-data")
        dataset_json_filepath = os.path.join(flat_inputs, "dataset.json")
        dataset_json = load_json(dataset_json_filepath)
        print(f"Looking for files ending on {dataset_json['file_ending']} in {flat_inputs}.")
        print(f"Found {len(os.listdir(flat_inputs))}")

        dataset_name = f"Dataset{int(dataset_id):03d}_XYZ".format(dataset_id)

        print("Dataset name:", dataset_name)
        nnunet_raw_dataset = os.path.join(nnunet_raw, dataset_name)
        print(f"Creating folder {nnunet_raw_dataset}.")
        Path(nnunet_raw_dataset).mkdir(exist_ok=True)
        nnunet_raw_dataset_imgs = os.path.join(nnunet_raw, dataset_name, "imagesTr")
        Path(nnunet_raw_dataset_imgs).mkdir(exist_ok=True)

        files = [f for f in os.listdir(flat_inputs) if f.endswith(dataset_json["file_ending"])]
        print(f"Found {len(files)} files ... Copying them to {nnunet_raw_dataset_imgs}.")
        # Move raw-data files over.
        for f in files:
            shutil.copy(os.path.join(flat_inputs, f), os.path.join(nnunet_raw_dataset_imgs, f))
        print(f"Moved {len(os.listdir(nnunet_raw_dataset_imgs))} files to {nnunet_raw_dataset_imgs}")
        shutil.copy(dataset_json_filepath, os.path.join(nnunet_raw_dataset, "dataset.json"))

    else:
        print("Not on valohai.")
        # Local paths are fine, no need to change anything.
        pass


def serialize_files_and_move_to_valohai_outputs(some_file_path: str, meta_data_dict: dict | None = None):
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


def get_all_file_in_dir(dir_path: str) -> list[str]:
    """Get all path files to the files in the directory and subdirectories."""
    files = []
    for f in Path(dir_path).iterdir():
        if f.is_file():
            if f.name not in [".DS_Store", "._.DS_Store"]:
                files.append(str(f))
            else:
                continue
        else:
            files += get_all_file_in_dir(f)
    return files


def save_files_on_valohai(path_to_copy: str, meta_data_dict: dict | None = None):
    """
    Takes all files that were written into the nnUNet_preprocessed folder
    and serializes them to the valohai output folder (if running in valohai).
    """
    if is_running_in_valohai():
        pp_path = path_to_copy
        all_files = get_all_file_in_dir(pp_path)
        for f in tqdm(all_files):
            serialize_files_and_move_to_valohai_outputs(f, meta_data_dict)
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
    os.environ["nnUNet_raw"] = "/home/tassilowald/Data/pseudo_valohai/pseudo_raw"
    os.environ["nnssl_preprocessed"] = "/home/tassilowald/Data/pseudo_valohai/pseudo_pp"
    os.environ["nnssl_results"] = "/home/tassilowald/Data/pseudo_valohai/pseudo_res"
    prepare_preprocessing_paths_on_valohai(1)
    save_files_on_valohai(os.environ["nnUNet_raw"], {"some": "meta_data"})
    print(nnUNet_raw)  # Make sure this is actually overwritten!
    print(nnssl_preprocessed)
    print("Done")
