from argparse import ArgumentParser
from pathlib import Path
import os
import re


def get_on_datasets(path: Path):
    # Get all datasets from openneuro
    datasets = []
    for dataset in os.listdir(path):
        if re.match(r"ds\d{6}", dataset):
            datasets.append(dataset)
    return datasets


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", type=Path, help="Path to openneuro")
    args = parser.parse_args()

    if Path(args.p).exists():
        # Get dataset in dir.
        ds = get_on_datasets(args.p)

    else:
        print("S3-Path not found")
