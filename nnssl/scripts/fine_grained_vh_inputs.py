from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import pandas as pd
from tqdm import tqdm
from valohai.config import is_running_in_valohai
import SimpleITK as sitk
import requests
import glob
from batchgenerators.utilities.file_and_folder_operations import save_json

MIN_FOV = (100, 100, 100)  # At least 10cm in each direction
MAX_SPACING = 5  # At most 3mm in any direction


def filter_mri_case(mri: Path, by_fov: bool = True, by_spacing: bool = True):
    """Filter MRI by field of view and spacing."""
    try:
        im = sitk.ReadImage(mri)
        spacing = im.GetSpacing()
        if len(spacing) != 3:
            return None
        fov = im.GetWidth() * spacing[0], im.GetHeight() * spacing[1], im.GetDepth() * spacing[2]
        if by_fov and any(f < MIN_FOV[i] for i, f in enumerate(fov)):
            return None
        if by_spacing and any(s >= MAX_SPACING for s in spacing):
            return None

        return mri
    except:
        return None


@dataclass
class ValohaiDataset:
    pure_t1: pd.DataFrame
    pure_t2: pd.DataFrame
    pure_flair: pd.DataFrame
    t1_flair: pd.DataFrame
    t2_flair: pd.DataFrame
    mra: pd.DataFrame


def get_strong_magnet_patients(data_df: pd.DataFrame) -> pd.DataFrame:
    strong_magnet_df = data_df[data_df["magneticfieldstrength"] >= 1.5]
    return strong_magnet_df


def get_subsets_of_interest(data_df: pd.DataFrame) -> ValohaiDataset:
    """
    Subset of interest for the FLOY project.
    """
    normal_t1 = data_df[(data_df["weighting"] == "T1") & (data_df["inversion_recovery"].isna())]
    t1_flair = data_df[(data_df["weighting"] == "T1") & (data_df["inversion_recovery"] == "FLAIR")]
    normal_t2 = data_df[(data_df["weighting"] == "T2") & data_df["inversion_recovery"].isna()]
    t2_flair = data_df[(data_df["weighting"] == "T2") & (data_df["inversion_recovery"] == "FLAIR")]
    just_flair = data_df[(pd.isna(data_df["weighting"])) & (data_df["inversion_recovery"] == "FLAIR")]
    mr_angio = data_df[data_df["weighting"].isin(["MRA", "MR Angiography"]) & (data_df["inversion_recovery"].isna())]
    return ValohaiDataset(
        pure_t1=normal_t1,
        pure_t2=normal_t2,
        pure_flair=just_flair,
        t1_flair=t1_flair,
        t2_flair=t2_flair,
        mra=mr_angio,
    )


def get_patients_from_df(data_df: pd.DataFrame, n: int | None = None) -> list[str]:
    """
    Choose the first n files from the meta data file
    :param meta_data_path:
    :param n: Get only the first n files (If None, get all)
    :return:
    """
    filenames = data_df["seriesinstanceuid"].tolist()
    if n is not None:
        filenames = filenames[:n]
    return filenames


def get_meta_data_df() -> pd.DataFrame:
    meta_dict_path: str
    if is_running_in_valohai():
        meta_dict_path = "/valohai/inputs/meta-data/full_meta.csv"
    else:
        meta_dict_path = "/home/tassilowald/Projects/FLOY/full_meta.csv"
    return pd.read_csv(meta_dict_path)


def get_series_dict() -> dict[str, str]:
    if is_running_in_valohai():
        all_series_ids = os.listdir("/valohai/inputs/all-data")
        series_dict = {p: os.path.join("/valohai/inputs/all-data", p) for p in all_series_ids}
    else:
        series_dict = {}
        for pat in Path("/home/tassilowald/Data/Datasets/mr-head-150").iterdir():
            if not pat.is_dir():
                continue
            for s in pat.iterdir():
                if s.name.endswith(".nii.gz"):
                    series_dict[s.name] = str(s)
    series_id_dict = {}
    for k, v in series_dict.items():
        if k.endswith(".nii.gz"):
            series_id_dict[k.split(".")[0]] = v
        else:
            series_id_dict[k] = v
    return series_id_dict


def main():

    # Series Dict contains series_UID to path to file.
    series_dict: dict[str, str] = get_series_dict()

    all_pats = get_meta_data_df()
    strong_magnet_pats = get_strong_magnet_patients(all_pats)
    valohai_dataset = get_subsets_of_interest(strong_magnet_pats)
    for key, dataset in valohai_dataset.__dict__.items():
        meta_data_json = {"valohai.dataset-versions": [f"dataset://fiona_preraw_{key}/v0"]}

        print(f"{key}: {len(dataset)}")
        pats = get_patients_from_df(dataset)
        rem_pats = [v for k, v in series_dict.items() if k in pats]
        print(f"Found {len(rem_pats)} of {len(pats)}")
        # Verify the MRI fullfills our spacing and FOV criteria.
        filtered_rem_pats = [
            rem_pat
            for rem_pat in tqdm(rem_pats, desc="Filtering MRIs by Spacing and FOV")
            if filter_mri_case(rem_pat) is not None
        ]
        if is_running_in_valohai():
            for frp in tqdm(filtered_rem_pats, desc="Copying over MRIs to output and tagging with metadata.json"):
                shutil.copy(frp, Path(f"/valohai/outputs/{frp.name}"))
                save_json(meta_data_json, frp.name + ".metadata.json")
        else:
            print("Just testing. Not running in Valohai.")


if __name__ == "__main__":
    main()
