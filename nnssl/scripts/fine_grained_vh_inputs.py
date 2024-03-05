from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import valohai_cli
import json
import requests


@dataclass
class ValohaiDataset:
    pure_t1_df: pd.DataFrame
    pure_t2_df: pd.DataFrame
    pure_flair_df: pd.DataFrame
    t1_flair_df: pd.DataFrame
    t2_flair_df: pd.DataFrame
    mra_df: pd.DataFrame


def get_full_patient_dataframe(path_to_csv: str = "/home/tassilowald/Projects/FLOY/full_meta.csv") -> pd.DataFrame:
    return pd.read_csv(path_to_csv)


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
        pure_t1_df=normal_t1,
        pure_t2_df=normal_t2,
        pure_flair_df=just_flair,
        t1_flair_df=t1_flair,
        t2_flair_df=t2_flair,
        mra_df=mr_angio,
    )


def get_patients_from_df(data_df: pd.DataFrame, n: int | None = None):
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


def main():
    all_pats = get_full_patient_dataframe()
    strong_magnet_pats = get_strong_magnet_patients(all_pats)
    valohai_dataset = get_subsets_of_interest(strong_magnet_pats)

    for key, dataset in valohai_dataset.__dict__.items():
        print(f"{key}: {len(dataset)}")
        pats = get_patients_from_df(dataset)

        s3_path = Path("s3://floy-data/clean-data/external/fiona/mr-head-full/")
        s3_filenames = [str(s3_path / (f + ".nii.gz")) for f in pats]

        post_url = "https://app.valohai.com/api/v0/dataset-versions/"

        post_request_body = {
            "name": f"fiona_preraw_{key}",
            "dataset": "018d5ae8-b4ae-2363-1e34-9a116fe8e800",
            "files": [{"datum": v} for v in s3_filenames],
        }

        ret = requests.post(post_url, post_request_body)



if __name__ == "__main__":
    main()
