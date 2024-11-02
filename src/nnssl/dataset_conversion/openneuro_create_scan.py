import glob
from pathlib import Path
import argparse
from batchgenerators.utilities.file_and_folder_operations import load_json, join, save_json, isfile
from tqdm import tqdm
import SimpleITK as sitk
from multiprocessing import Pool, cpu_count
import os
from nnssl.paths import nnssl_raw
import signal


local_openneuro_dir = join(os.environ["E132Rohdaten"] + "/WaitingRoom/mic_rocket/raw/openneuro_full_v2")


def main():

    final_scan_list = []
    lists_in_list = []

    dataset_dirs = sorted(Path(local_openneuro_dir).iterdir())

    with Pool(processes=cpu_count()) as pool:
        scan_lists = pool.map(_task, dataset_dirs)
    scan_lists = [n for n in scan_lists if n]

    if scan_lists:
        print(f"Gathered {len(scan_lists)} new scan list(s)!")
    else:
        print("No new scan lists needed!")

    for sl in scan_lists:
        lists_in_list.append(sl)

    lists_in_list.sort(key=lambda sl: sl[0]["path"])

    for sl in lists_in_list:
        final_scan_list.extend(sl)

    save_json(final_scan_list, join(nnssl_raw, "Dataset744_OpenNeuro", "scans_ALL.json"), sort_keys=False)

    chunks = []
    for i, chunk in enumerate(split(final_scan_list, num_chunks=20)):
        chunks.append(chunk)
        save_json(chunk, join(nnssl_raw, "Dataset744_OpenNeuro", f"scans_{i:02d}.json"), sort_keys=False)

    print(f"Successfully gathered all {len(final_scan_list)} scans!")


def split(lis, num_chunks):
    k, m = divmod(len(lis), num_chunks)
    return [lis[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(num_chunks)]


def _task(dataset_dir):
    scan_list = []
    try:
        scan_metadata_json = load_json(join(dataset_dir, "scan_metadata.json"))
        subject_metadata_json = load_json(join(dataset_dir, "subject_metadata.json"))
    except:
        return []

    subject_subjects_dict = subject_metadata_json["subjects"]

    ds_id = dataset_dir.name
    if ds_id != scan_metadata_json["dataset_id"] or ds_id != subject_metadata_json["dataset_id"]:
        raise ValueError(
            f"{str(dataset_dir.resolve())}: non-matching dataset IDs:"
            f"\n\tdataset ID: {ds_id}"
            f"\n\tscan_metadata_json: {scan_metadata_json["dataset_id"]}"
            f"\n\tsubject_metadata_json: {subject_metadata_json["dataset_id"]}"
        )

    reader = sitk.ImageFileReader()

    for sub_id, sub in scan_metadata_json["subjects"].items():
        for scan_path, scan_metadata in sub.items():
            scan_path = join(ds_id, sub_id, scan_path)

            # parts = scan_path.split(os.path.sep)
            # data_type, scan_name = parts[-2], parts[-1]

            full_scan_path = join(local_openneuro_dir, scan_path)

            try:
                reader.SetFileName(full_scan_path)
                reader.ReadImageInformation()
                size = reader.GetSize()
                spacing = reader.GetSpacing()
                dimensions = reader.GetDimension()
            except Exception as e:
                print(f"unknown exception for {scan_path}:")
                print(repr(e))
                spacing = None
                dimensions = None
                size = None
                continue

            scan_dict = {
                "path": scan_path,
                "spacing": spacing,
                "dimension": dimensions,
                "size": size,
                **subject_subjects_dict[sub_id],
                **scan_metadata,
            }
            del scan_dict["num_scans"]
            scan_list.append(scan_dict)

    # save, even is empty
    # Path(join(nnssl_raw, "Dataset744_OpenNeuro", "jsons")).mkdir(exist_ok=True, parents=True)
    # save_json(scan_list, (nnssl_raw, "Dataset744_OpenNeuro", "jsons", ds_id + ".json"))

    return scan_list


if __name__ == "__main__":
    main()
