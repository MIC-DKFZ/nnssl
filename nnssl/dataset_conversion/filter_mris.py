from pathlib import Path
from functools import partial
import shutil
import SimpleITK as sitk
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json

from nnssl.dataset_conversion.filter_mris_all import filter_mri_case

MIN_FOV = (100, 100, 100)  # At least 10cm in each direction
MAX_SPACING = 3  # At most 3mm in any direction


def get_meta_data_of_pat(mri_name: str, meta_data_df: pd.DataFrame, pat_map: dict) -> dict:
    """Get meta data of patient."""
    pat_id = mri_name.split("__")[0].split("_")[-1]
    mri_id = mri_name.split("__")[1].split("_")[-1]
    series_id = pat_map[pat_id]["images"][mri_id].split(".")[0]
    pat_df = meta_data_df[meta_data_df["seriesinstanceuid"] == series_id].to_dict("records")[0]
    return pat_df


def helper(func):
    def parallel_func(args):
        return func(*args)

    return parallel_func


def filter_mris(
    mris: list[Path],
    pat_map: dict,
    meta_data_df: pd.DataFrame,
    by_fov: bool = True,
    by_spacing: bool = True,
    by_meta_info: bool = True,
    n_proc: int = 1,
):
    """Filter MRIs by field of view and spacing."""
    # ------------------------------ Meta Data here ------------------------------ #
    meta_data = [get_meta_data_of_pat(mri.name, meta_data_df, pat_map) for mri in mris]

    partial_filter = partial(filter_mri_case, by_fov=by_fov, by_spacing=by_spacing, by_meta_info=by_meta_info)

    if n_proc == 1:
        rem_mris = [partial_filter(mri, meta_info) for mri, meta_info in tqdm(zip(mris, meta_data), total=len(mris))]
    else:
        with ProcessPoolExecutor(max_workers=n_proc) as executor:
            rem_mris = list(tqdm(executor.map(helper(partial_filter), mris), total=len(mris)))
    rem_mris = [mri for mri in rem_mris if mri is not None]
    return rem_mris


def save_mris(mris: list[Path], out_dir: Path):
    """Save filtered MRIs to out_dir."""
    for mri in mris:
        shutil.copy(mri, out_dir / mri.name)


if __name__ == "__main__":
    mris_dir = Path("/home/tassilowald/Data/Datasets/nnunetv2/nnssl_raw/Dataset737_FloyPrototype")
    out_dir = Path("/home/tassilowald/Data/Datasets/nnunetv2/nnssl_raw/Dataset739_FloyPrototype_more_filtered")
    meta_data_df: pd.DataFrame = pd.read_csv("/home/tassilowald/Data/Datasets/mr-head-150/mr_150_meta.csv")
    pat_map: dict = load_json(
        "/home/tassilowald/Data/Datasets/nnunetv2/nnssl_raw/Dataset737_FloyPrototype/patient_id_mapping.json"
    )
    out_dir.mkdir(exist_ok=True)
    mris = list(mris_dir.glob("**/*.nii.gz"))
    n_mris = len(mris)
    dataset_json = {
        "channel_names": {"0": "someMRI"},
        "file_ending": ".nii.gz",
        "numTraining": n_mris,
        "name": "Cases of Floys initial Dataset",
        "release": "0.0",
        "licence": "Proprietary -- do not touch without permission",
        "description": "Unlabeled set of datapoints that are used for pre-text task pretraining",
    }
    save_json(dataset_json, out_dir / "dataset.json")
    train_path = out_dir / "imagesTr"
    train_path.mkdir(exist_ok=True)
    filtered_mris = filter_mris(mris, pat_map, meta_data_df, n_proc=1)
    save_mris(filtered_mris, train_path)
