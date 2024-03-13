from pathlib import Path
import argparse


from nnssl.dataset_conversion.filter_mris_all import filter_mri_case
from nnssl.scripts.fine_grained_vh_inputs import get_valohai_series_dict
from nnssl.scripts.valohai_requests import maybe_create_new_dataset_version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()
    ver = args.version
    ds_name = args.dataset_name

    ingested_files_json = get_valohai_series_dict("raw-data")
    dataset_Path = Path("/valohai/inputs/raw-data")

    cases_fullfilling_citeria: list[Path] = []
    datum_uids: list[str] = []
    for scan in dataset_Path.iterdir():
        if scan.name.endswith(".nii.gz"):
            if filter_mri_case(scan) is None:
                continue
            cases_fullfilling_citeria.append(scan)
            datum_uids.append(ingested_files_json[scan.name.split(".")[0]]["datum_id"])

    dataset_name = f"fiona_filtered_{ds_name}"
    print(f"Data in new dataset: {len(cases_fullfilling_citeria)} of {len(ingested_files_json)}")
    owner = 5425
    req_resp = maybe_create_new_dataset_version(dataset_name, version=ver, files=datum_uids, owner=owner)


if __name__ == "__main__":
    main()
