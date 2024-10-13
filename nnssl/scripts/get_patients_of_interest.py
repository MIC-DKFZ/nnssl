import pandas as pd
import requests
from nnssl.scripts.fine_grained_vh_inputs import (
    get_meta_data_df,
    get_patients_with_meta_data_from_df,
    get_strong_magnet_patients,
    get_subsets_of_interest,
    get_patients_from_df,
)


from nnssl.scripts.valohai_requests import (
    convert_andrei_adtop_to_lookup,
    get_andrei_adopt_output,
    get_auth_header,
    get_datum_uids_in_dataset_content,
    get_dataset_versions,
    get_name_from_datum_uid,
    maybe_create_new_dataset_version,
)
from loguru import logger


DATASETS_USED = [
    "018e765d-442d-97e5-d561-a744fd75168c",  # PURE T2
    "018e39ac-432d-19b3-f0a5-4effab28e258",  # PURE T1
    "018e76c5-8ddc-dee9-5632-2abe5e6e47af",  # PURE FLAIR
    "018e76c7-9890-f89c-8dd5-83ba23ef31c0",  # T1 FLAIR
    "018e76c7-b75c-8d43-434a-0b4cb46d11b4",  # T2 FLAIR
]

PURE_T2_VERSIONS = [
    # From T2: 018e765d-442d-97e5-d561-a744fd75168c,
    "018e765d-5d02-dab4-9900-46811fe86b2f",
    "018e76c1-0c70-59e6-a121-f33976ddd5f1",
    "018e76c1-16ad-4fd0-259a-d694ac86a030",
    "018e76c1-0950-e568-09c7-83c025d2e61d",
    "018e76c2-3c1f-fecb-3ea9-debbaa825958",
    "018e76c2-3520-6571-dba3-815a42dd32f9",
    "018e76c2-2e44-5f87-d5cc-bdc332e48ddc",
    "018e76c3-3109-23e7-6cb4-0d118c43a236",
    "018e76c3-34ac-7d12-93e9-aad099b89332",
    "018e76c4-48b4-1c09-2cd8-269b23435957",
    "018e76c4-4131-8387-84d0-ca434f04137c",
    "018e76c4-e60f-1e68-12eb-ec5dbce927b0",
    "018e76c3-8a72-ee1d-ebfd-1e551829a915",
]

PURE_T1_VERSIONS = [
    # 018e39ac-432d-19b3-f0a5-4effab28e258,
    "018e39ac-5fd1-afa7-7a83-820d3d89a242",
    "018e39ac-6642-310c-6ac7-de0fbb36d087",
    "018e39ac-a6dc-5d62-e523-3ff7bc478f95",
    "018e39af-672f-eb7e-8c71-22e04e1d4505",
    "018e39af-61cd-156c-7601-63bf30bed3c0",
    "018e39ae-c5f4-cd0d-16f8-80020668109c",
    "018e39af-1ad6-754a-38be-afb559ee5361",
    "018e39af-2005-2dd0-5c30-eb6c43ba4ec8",
    "018e39b1-1b3a-f37f-533a-2361250ade78",
    "018e39b1-b848-ac2f-1c25-1db040d45b76",
    "018e39b1-d2d5-e039-c37a-808d0cfff5bd",
    "018e39b1-f79e-39d2-ac86-4489440d2538",
    "018e39b1-d6b3-6126-15c3-a359f297cd22",
    "018e39b3-b1b3-7f5b-a036-70caa4317d16",
    "018e39b4-38da-1d12-30aa-f9767ad95cb0",
    "018e39b4-b6a4-3738-3c8a-757304b3cfba",
    "018e39b4-46f6-78f9-b00b-0484af2c3061",
    "018e39b4-c316-dbca-f6f9-35a6dec50f51",
    "018e39b6-79fc-1a3e-71c9-b0b295c20d93",
    "018e39b6-808f-9eca-73c6-2384caacc37e",
    "018e39b5-36ab-5a1c-5730-1e8b9e0ac18f",
]

PURE_FLAIR_VERSIONS = [
    "018e76c5-ad9e-9c8c-1006-551984a59733",
    "018e76c5-b59b-1e9d-2014-8400451e0d75",
    "018e76c6-9c44-82c5-91ea-c702ff8fa69e",
    "018e76c6-88fe-559d-107c-97adcd924ce6",
    "018e76c6-ddd4-6f92-c802-6e6301b2ec54",
    "018e76c7-accb-6c1f-9a51-26ffc098db68",
    "018e76c7-bc9c-2b37-eed9-1a206684d9af",
    "018e76c9-c220-68f6-3846-b6997582c2d1",
    "018e76c7-fe08-0f47-c67e-c1cf35dd296d",
]

T1_FLAIR = [
    # 018e76c7-9890-f89c-8dd5-83ba23ef31c0,
    "018e76c7-b42f-96c5-f394-ef0b83f6d0ae"
]

T2_FLAIR = [
    # 018e76c7-b75c-8d43-434a-0b4cb46d11b4
    "018e76c9-0ce5-d038-1158-d550aa0bea06",
    "018e76c7-c2e8-33dd-ae66-1158fb59af92",
]


all_dataset_version_combos = [
    ("018e765d-442d-97e5-d561-a744fd75168c", PURE_T2_VERSIONS),
    ("018e39ac-432d-19b3-f0a5-4effab28e258", PURE_T1_VERSIONS),
    ("018e76c5-8ddc-dee9-5632-2abe5e6e47af", PURE_FLAIR_VERSIONS),
    ("018e76c7-9890-f89c-8dd5-83ba23ef31c0", T1_FLAIR),
    ("018e76c7-b75c-8d43-434a-0b4cb46d11b4", T2_FLAIR),
]


def main():
    all_pats = get_meta_data_df()

    all_patients_used = []
    for ds, versions in all_dataset_version_combos:
        for version in versions:
            all_patients_used.extend(get_datum_uids_in_dataset_content(version))
    patients_seriesinstance_uids = [pat_used["name"].split("/")[1].replace(".nii.gz", "") for pat_used in all_patients_used]

    used_pats = all_pats[all_pats["seriesinstanceuid"].isin(patients_seriesinstance_uids)]
    used_pats.to_csv("used_pats.csv")

    print("Wait")


def old_main():
    anreis_adopt_outputs = get_andrei_adopt_output()
    lookup_for_pats = convert_andrei_adtop_to_lookup(andrei_adopt_output=anreis_adopt_outputs)

    all_pats = get_meta_data_df()
    strong_magnet_pats = get_strong_magnet_patients(all_pats)
    valohai_dataset = get_subsets_of_interest(strong_magnet_pats)
    all_patients = []
    for k, v in valohai_dataset.__dict__.items():
        logger.info(f"Working on {k}")
        pats = get_patients_with_meta_data_from_df(v)  # To fill the datum_uids

        for p_name, p_vals in pats.items():
            if p_name in lookup_for_pats:
                all_patients.append(p_vals)
                # all_patients.append(**p + {"datum": lookup_for_pats[p]["id"], "name": })
        # dataset_name = f"fiona_full_{k}"
        # version = "v1"
        # owner = 5425  # Floys org id

        # n_files = len(files)
        # n_batches = ((n_files // 1000) + 1) if (n_files % 1000) != 0 else n_files // 1000
        # for i in range(n_batches):
        #     start = i * 1000
        #     end = (i + 1) * 1000 if (i != (n_batches - 1)) else -1
        #     req_resp = maybe_create_new_dataset_version(
        #         dataset_name, version=version + f"_part_{i}", files=files[start:end], owner=owner
        #     )
        #     try:
        #         response_message = req_resp.json()
        #     except AttributeError:
        #         response_message = "No message in response"
        #     logger.info(f"Response message: {response_message}")
    all_patients = pd.DataFrame(all_patients)
    all_patients.to_csv("all_patients.csv")


if __name__ == "__main__":
    main()
