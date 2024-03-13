from nnssl.scripts.fine_grained_vh_inputs import (
    get_meta_data_df,
    get_strong_magnet_patients,
    get_subsets_of_interest,
    get_patients_from_df,
)


from nnssl.scripts.valohai_requests import (
    convert_andrei_adtop_to_lookup,
    get_andrei_adopt_output,
    maybe_create_new_dataset_version,
)
from loguru import logger


def main():
    anreis_adopt_outputs = get_andrei_adopt_output()
    lookup_for_pats = convert_andrei_adtop_to_lookup(andrei_adopt_output=anreis_adopt_outputs)

    all_pats = get_meta_data_df()
    strong_magnet_pats = get_strong_magnet_patients(all_pats)
    valohai_dataset = get_subsets_of_interest(strong_magnet_pats)
    for k, v in valohai_dataset.__dict__.items():
        logger.info(f"Working on {k}")
        pats = get_patients_from_df(v)  # To fill the datum_uids

        files = []
        for p in pats:
            if p in lookup_for_pats:
                files.append({"datum": lookup_for_pats[p]["id"]})
        dataset_name = f"fiona_full_{k}"
        version = "v0"
        owner = 5425  # Floys org id
        req_resp = maybe_create_new_dataset_version(dataset_name, version=version, files=files, owner=owner)

    return


if __name__ == "__main__":
    main()
