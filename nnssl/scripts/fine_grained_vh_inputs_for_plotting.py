from loguru import logger
import pandas as pd
from tqdm import tqdm
from nnssl.dataset_conversion.filter_mris_all import filter_mri_case
from nnssl.scripts.fine_grained_vh_inputs import (
    create_local_series_dict,
    get_meta_data_df,
    get_mr150_data_df,
    get_patients_from_df,
    get_strong_magnet_patients,
    get_subsets_of_interest,
    get_valohai_series_dict,
)
from valohai.config import is_running_in_valohai
from batchgenerators.utilities.file_and_folder_operations import save_json


def main():
    # Series Dict contains series_UID to path to file.
    logger.info("Starting to create Valohai inputs.")
    if is_running_in_valohai():
        data_id_to_info_json: dict[str, dict] = get_valohai_series_dict("all-data")
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

    all_ids = []
    all_pats_out = []

    for key, val in valohai_dataset.__dict__.items():
        logger.info(f"Working on {key}")
        pats = get_patients_from_df(val)

        # For all pats read from the csv file, check if they are in the valohai dataset.
        for pat in tqdm(pats, desc=f"{key}: Checking if cases are present and fulfill criteria."):
            if pat in data_id_to_info_json:
                # If the MRI is in the present dataset, check if it fulfills our criteria.

                if filter_mri_case(data_id_to_info_json[pat]["path"]) is not None:
                    all_ids.append(data_id_to_info_json[pat])
                    all_pats_out.append(pat)

    if is_running_in_valohai():
        save_json(all_ids, f"/valohai/outputs/all_ids.json")
        save_json(all_pats_out, f"/valohai/outputs/all_pats.json")

    return


if __name__ == "__main__":
    main()
