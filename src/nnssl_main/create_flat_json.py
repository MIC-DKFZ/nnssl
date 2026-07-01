from pathlib import Path

from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
from tqdm import tqdm
from nnssl.data.raw_dataset import Collection
import pandas as pd

col_order = [
    "unique_id",
    "modality",
    "age",
    "weight",
    "bmi",
    "sex",
    "handedness",
    "race",
    "health_status",
    "manufacturer",
    "model_name",
    "phase_encoding_direction",
    "repetition_time",
    "echo_time",
    "is_brain_extract",
    "derived_from"
]

def main():
    output_dir = Path("/home/j385i/cluster_data/j385i/data/openneuro/")
    pretrain_json = load_json("/home/j385i/cluster_data/j385i/datasets/nnssl/nnssl_raw/Dataset745_OpenNeuro_v2/pretrain_data.json")
    collection = Collection.from_dict(pretrain_json)
    independent_images = collection.to_independent_images()
    flat_json = []
    for ind_image in tqdm(independent_images):
        unique_id = ind_image.get_unique_id()[25:]
        flat_json.append(
            {
                "unique_id": unique_id,
                "image_path": ind_image.image_path,
                "modality": ind_image.image_modality,
                **ind_image.subject_info,
                **ind_image.image_info
            }
        )
    save_json(flat_json, output_dir / "openneuro_flat.json")

    # flat_json = load_json("openneuro_flat.json")
    df = pd.DataFrame(flat_json)
    df.drop(columns=["image_path"])
    df = df[col_order]
    df.to_csv(output_dir / "openneuro_flat.csv", index=False)


if __name__ == "__main__":
    main()