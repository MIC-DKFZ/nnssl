from nnssl.scripts.fine_grained_vh_inputs import get_meta_data_df
from nnssl.scripts.valohai_requests import get_execution_output, get_valohai_projects, get_valohai_stores


def main():
    execution_id = "018e1622-540c-4ed6-ed44-72c3194758a7"
    projs = get_valohai_projects()
    stores = get_valohai_stores()
    all_pats = get_execution_output(execution_id)
    # all_csvs =
    # create_dataset_from_execution_output(all_pats)

    pre_filter_patients = [ap["name"] for ap in all_pats]
    pre_filter_patients = [ap for ap in pre_filter_patients if ap.endswith(".nii.gz")]
    patients_seriesinstance_uids = [pat_used.split("/")[1].replace(".nii.gz", "") for pat_used in pre_filter_patients]

    all_pats = get_meta_data_df()

    used_pats = all_pats[all_pats["seriesinstanceuid"].isin(patients_seriesinstance_uids)]
    used_pats.to_csv("pre_filter_pats.csv")

    data_to_pat_uuids: dict[str, str] = {}
    for pat in all_pats:
        _query: dict[str, str] = {
            "wildcard_url": pat["url"],
            "id": pat["id"],
            "store": "floy-gcp",
            "project": "MR-HEAD",
            "name": pat["name"],
        }
        query = {
            "name": pat["name"],
            "store": stores["floy-gcp"]["id"],
            "project": projs["MR-Head"]["id"],
            "output_execution": execution_id,
        }
        # uuid = check_for_datum_uuid(query)
        # if uuid is not None:
        #     data_to_pat_uuids()
    return


if __name__ == "__main__":
    main()
