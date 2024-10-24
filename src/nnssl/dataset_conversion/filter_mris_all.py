from pathlib import Path
from functools import partial
import shutil
import SimpleITK as sitk
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
import numpy as np


def filter_mri_case(
    mri: Path, meta_info: dict | None = None, by_fov: bool = True, by_spacing: bool = True, by_meta_info: bool = True
):
    """Filter MRI by field of view and spacing."""
    MIN_FOV = (50, 50, 50)  # At least 5cm in each direction
    MAX_SPACING = 6.5  # At most 6.5mm in any direction
    try:
        im = sitk.ReadImage(mri)
        file_size_kb = Path(mri).stat().st_size / 1024
        # If the file is smaller than 200kb, it is probably broken
        if file_size_kb < 200:
            return "File too small"

        spacing = im.GetSpacing()
        if len(spacing) != 3:
            return "Spacing not 3D"
        fov = im.GetWidth() * spacing[0], im.GetHeight() * spacing[1], im.GetDepth() * spacing[2]
        if by_fov and any(f < MIN_FOV[i] for i, f in enumerate(fov)):
            return "FOV too small"
        if by_spacing and any(s >= MAX_SPACING for s in spacing):
            return "Spacing too large"
        if by_meta_info and (meta_info is not None):
            series_description_exclusions = ["adc", "pha", "dwi"]
            if not pd.isna(meta_info["weighting"]):
                if "SWI" in str(meta_info["weighting"]):
                    return "Weighted SWI"
            if not pd.isna(meta_info["seriesdescription"]):
                if any([sde in meta_info["seriesdescription"] for sde in series_description_exclusions]):
                    return "Series description exclusion"

        return Path(mri)
    except:
        return None


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
    rem_mris = [mri for mri in rem_mris if isinstance(mri, Path)]
    return rem_mris


def save_mris(mris: list[Path], out_dir: Path):
    """Save filtered MRIs to out_dir."""
    for mri in mris:
        shutil.copy(mri, out_dir / mri.name)


if __name__ == "__main__":
    meta_data_df: pd.DataFrame = pd.read_csv("/home/tassilowald/Projects/FLOY/full_meta.csv")
    strong_magnet_df = meta_data_df[meta_data_df["magneticfieldstrength"] >= 1.5]

    normal_t1 = strong_magnet_df[
        (strong_magnet_df["weighting"] == "T1") & (strong_magnet_df["inversion_recovery"].isna())
    ]
    t1_flair = strong_magnet_df[
        (strong_magnet_df["weighting"] == "T1") & (strong_magnet_df["inversion_recovery"] == "FLAIR")
    ]
    normal_t2 = strong_magnet_df[
        (strong_magnet_df["weighting"] == "T2") & strong_magnet_df["inversion_recovery"].isna()
    ]
    t2_flair = strong_magnet_df[
        (strong_magnet_df["weighting"] == "T2") & (strong_magnet_df["inversion_recovery"] == "FLAIR")
    ]
    just_flair = strong_magnet_df[
        (pd.isna(strong_magnet_df["weighting"])) & (strong_magnet_df["inversion_recovery"] == "FLAIR")
    ]
    mr_angio = strong_magnet_df[
        strong_magnet_df["weighting"].isin(["MRA", "MR Angiography"])
        & (strong_magnet_df["inversion_recovery"].isna())
    ]
    pat_map: dict = load_json(
        "/home/tassilowald/Data/Datasets/nnunetv2/nnssl_raw/Dataset737_FloyPrototype/patient_id_mapping.json"
    )

    # filtered_mris = filter_mris(mris, pat_map, meta_data_df, n_proc=1)
    # save_mris(filtered_mris, train_path)
