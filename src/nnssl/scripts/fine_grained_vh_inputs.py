from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from valohai.config import is_running_in_valohai
from batchgenerators.utilities.file_and_folder_operations import save_json, load_json
from loguru import logger

from nnssl.dataset_conversion.filter_mris_all import filter_mri_case


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


def get_patients_with_meta_data_from_df(data_df: pd.DataFrame, n: int | None = None) -> dict[str:dict]:
    """
    Choose the first n files from the meta data file
    :param meta_data_path:
    :param n: Get only the first n files (If None, get all)
    :return:
    """
    filenames = data_df["seriesinstanceuid"].tolist()
    data_df = data_df.to_dict(orient="records")
    if n is not None:
        filenames = filenames[:n]
    return {f: data_df[i] for i, f in enumerate(filenames)}


def get_meta_data_df() -> pd.DataFrame:
    meta_dict_path: str
    if is_running_in_valohai():
        meta_dict_path = "/valohai/inputs/meta-data/full_meta.csv"
    else:
        meta_dict_path = "/home/tassilowald/Projects/FLOY/full_meta.csv"
    return pd.read_csv(meta_dict_path)


def get_mr150_data_df() -> pd.DataFrame:
    meta_dict_path: str
    if is_running_in_valohai():
        meta_dict_path = "/valohai/inputs/meta-data/full_meta.csv"
    else:
        meta_dict_path = "/home/tassilowald/Projects/FLOY/mr_150_meta.csv"
    return pd.read_csv(meta_dict_path)


def create_local_series_dict() -> dict[str, dict]:
    local_file_dict = {}
    for pat in Path("/home/tassilowald/Data/Datasets/mr-head-150").iterdir():
        if not pat.is_dir():
            continue
        for s in pat.iterdir():
            if s.name.endswith(".nii.gz"):
                local_file_dict[s.name.split(".")[0]] = {"path": str(s), "datum_id": None, "name": s.name}
    return local_file_dict


# def find_all_files_recursively(meta_json: dict)-> dict[str, dict]:
#     """
#     Find all files recursively in the dataset. This is used to create the valohai dataset.
#     """
#     all_files = {}
#     keys = meta_json.keys()
#     if "files" in keys:
#         for f in meta_json["files"]:
#             filename = f["name"].split(".")[0]
#             if filename.endswith(".nii.gz"):
#                 all_files[filename] = f  # All meta infos.
#     else:
#         if isinstance(meta_json, dict):
#             find_all_files_recursively(meta_json)

#     for k, v in meta_json.items():
#         if isinstance(v, dict):
#             all_files += find_all_files_recursively(v)
#         elif isinstance(v, list):
#             for i in v:
#                 all_files += find_all_files_recursively(i)
#         else:
#             if k == "path":
#                 all_files.append(v)
#     return all_files


def get_valohai_series_dict(dataset_name: str = "dataset") -> dict[str, dict]:
    """
    Returns a mapping of valohai file names to all meta information as provided in System configuration inputs.json
    https://docs.valohai.com/hc/en-us/articles/18704309491473-System-Configuration-Files
    """
    inputs_json = load_json("/valohai/config/inputs.json")
    print(inputs_json)  # Just for logging purposes.
    data_of_choice: list[dict] = inputs_json[dataset_name]["files"]
    local_file_id_dict = {}
    for data in data_of_choice:
        name = data["name"]
        if name.endswith(".nii.gz"):  # Only if it's a file of iterest.
            local_file_id_dict[data["name"].split(".")[0]] = data  # All meta infos.
    return local_file_id_dict


def main():
    # Series Dict contains series_UID to path to file.
    logger.info("Starting to create Valohai inputs.")
    if is_running_in_valohai():
        data_id_to_info_json: dict[str, dict] = get_valohai_series_dict()
    else:
        data_id_to_info_json = create_local_series_dict()

    all_pats: pd.DataFrame = get_meta_data_df()
    pats_150: pd.DataFrame = get_mr150_data_df()

    if not is_running_in_valohai():
        logger.info("Checking for differences between the 150 patients and the full dataset.")
        pats_150_series = set(pats_150["seriesinstanceuid"].tolist())
        all_pats_series = set(all_pats["seriesinstanceuid"].tolist())
        set_diff = pats_150_series.difference(all_pats_series)
        set_inter = pats_150_series.intersection(all_pats_series)
        logger.info(f"Set diff: {len(set_diff)}")
        logger.info(f"Set inter: {len(set_inter)}")

    strong_magnet_pats = get_strong_magnet_patients(all_pats)
    valohai_dataset = get_subsets_of_interest(strong_magnet_pats)

    n_total_used = 0
    n_total = len(data_id_to_info_json)
    for key, val in valohai_dataset.__dict__.items():
        logger.info(f"Working on {key}")
        pats = get_patients_from_df(val)
        all_files = []
        all_names = []
        dataset_uuids_of_ids = {"name": key, "dataset": "SomeID", "files": all_files, "file_names": all_names}
        names_for_local_running = {
            "name": key,
            "dataset": "SomeID",
            "files": all_names,
        }

        n_diff = 0
        n_in_both_sets = 0
        n_neither = 0
        # For all pats read from the csv file, check if they are in the valohai dataset.
        for pat in tqdm(pats, desc=f"{key}: Checking if cases are present and fulfill criteria."):
            if pat in data_id_to_info_json:
                # If the MRI is in the present dataset, check if it fulfills our criteria.
                if False:  # not is_running_in_valohai():
                    cur_name = data_id_to_info_json[pat]["name"].split(".")[0]
                    if cur_name in set_diff:
                        # logger.info(f"{cur_name} is only in the 150 dataset, not in the full on")
                        n_diff += 1
                    elif cur_name in set_inter:
                        # logger.info(f"{cur_name} is in both.")
                        n_in_both_sets += 1
                    else:
                        # logger.info(f"{cur_name} is not in either set.")
                        n_neither += 1

                if filter_mri_case(data_id_to_info_json[pat]["path"]) is not None:
                    all_files.append({"datum": data_id_to_info_json[pat]["datum_id"]})
                    all_names.append(data_id_to_info_json[pat]["name"])

        logger.info(f"{len(all_files)/(len(pats)+1e-9):.2%} of the cases fulfill the criteria.")
        # logger.info(f"Cases diff: {n_diff}")
        # logger.info(f"Cases inter: {n_in_both_sets}")
        # logger.info(f"Cases neither: {n_neither}")
        logger.info(f"Using {len(all_files)} cases for {key} of {len(pats)}.")
        n_total_used += len(all_files)
        if is_running_in_valohai():
            save_json(dataset_uuids_of_ids, f"/valohai/outputs/{key}.json")
        else:
            # print(json.dumps(dataset_uuids_of_ids, indent=4))
            # print(json.dumps(names_for_local_running, indent=4))
            pass
    logger.info(f"Used {n_total_used / n_total:.2%} of all data. {n_total_used} of {n_total} cases.")

    return


if __name__ == "__main__":
    main()
