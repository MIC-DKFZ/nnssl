from pathlib import Path

import pandas as pd
from template_brain_preprocessing import *


if __name__ == "__main__":
    out_dir = "/deepstore/datasets/mia/HealthyAI/nnssl_data/classification/preprocessed/abide_1mm_cropped_160_new"
    raw_data_dir = "/deepstore/datasets/mia/HealthyAI/nnUNet_raw_data/classification/raw/abide_1mm_cropped_160_new"
    base_dir = "/deepstore/datasets/mia/HealthyAI/ABIDE/ABIDE_img"
    csv_path = "/deepstore/datasets/mia/HealthyAI/ABIDE/Abide.csv"

    # 1. find all unique subject IDs
    subject_ids = [
        d.name.split("_")[1] for d in Path(base_dir).iterdir() if d.is_dir()
    ]  # BNI_29007_1 -> 29007 or UCLA_23424_followup -> 23424 or USMOSDC_234234_baseline_rf_rferf -> 234234
    # nii_files = list(Path(base_dir).rglob('*.nii.gz'))
    print(subject_ids)

    # 3. Load label CSV
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # clean up header whitespace
    df["Label"] = df["Group"].apply(lambda x: 1 if str(x).strip() == "Autism" else 0)

    # remove the hosptial code from subject IDs in the df
    df["Subject"] = (
        df["Subject"].str.split("_").str[1]
    )  # e.g., BNI_29007_1 -> 29007_1 or UCLA_23424_followup -> 23424_followup

    # 4. Map Subject ID to Label
    label_dict_full = dict(zip(df["Subject"].astype(str), df["Label"]))

    maybe_mkdir_p(join(raw_data_dir, "imagesTr"))
    # 5. Build label dict with full path as key
    label_dict = {}

    ###copy data in expected nnU-Net format
    for i, case_name in enumerate([d for d in Path(base_dir).iterdir() if d.is_dir()]):
        subject_id = str(case_name).split("/")[-1].split("_")[1]

        classification = label_dict_full.get(subject_id, None)

        if classification is None:
            print(f"Warning: Subject {subject_id} not found in label CSV.")
            continue

        full_file_name = list(case_name.rglob("anat.nii.gz"))
        if not full_file_name:
            print(f"Warning: No file found for subject {subject_id}.")
            continue

        label_dict[subject_id] = classification
        full_file_name = full_file_name[0]  # take the first match

        print(subject_id, label_dict[subject_id], full_file_name)

        img = sitk.ReadImage(join(full_file_name))
        sitk.WriteImage(
            img, join(raw_data_dir, "imagesTr", subject_id + "_0000.nii.gz")
        )

    maybe_mkdir_p(out_dir)
    save_json(label_dict, join(out_dir, "labels.json"))

    # predict brainmasks
    hd_bet_predict(raw_data_dir)
    load_crop_brainextract_normalize_images(
        raw_data_dir,
        out_dir,
        [1.0, 1.0, 1.0],
        [160, 160, 160],
        brain_extract=True,
        num_workers=4,
    )
