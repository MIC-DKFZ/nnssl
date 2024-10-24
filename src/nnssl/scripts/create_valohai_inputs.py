from pathlib import Path
import pandas as pd


def choose_first_n_files_from_meta_data(meta_data_path: Path, n: int):
    """
    Choose the first n files from the meta data file
    :param meta_data_path:
    :param n:
    :return:
    """
    meta_data_df = pd.read_csv(meta_data_path)
    long_files = meta_data_df[meta_data_df["serieslength"] >= 100]
    long_file_names = long_files["seriesinstanceuid"].tolist()
    return long_file_names[:n]


def main():
    meta_data_path = Path("/home/tassilowald/Projects/FLOY/full_meta.csv")
    sample_filenames = choose_first_n_files_from_meta_data(meta_data_path, 300)

    s3_path = Path("s3://floy-data/clean-data/external/fiona/mr-head-full/")
    s3_filenames = [str(s3_path / (f + ".nii.gz")) for f in sample_filenames]

    post_url = "https://app.valohai.com/api/v0/dataset-versions/"

    post_request_body = {
        "name": "v1",
        "dataset": "018d5ae8-b4ae-2363-1e34-9a116fe8e800",
        "files": [{"datum": v} for v in s3_filenames],
    }
    import requests
    requests.post(post_url, post_request_body)


if __name__ == "__main__":
    main()
