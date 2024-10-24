from functools import partial
import os
from pathlib import Path
from typing import Union
import multiprocessing

import numpy as np
from tqdm import tqdm

from nnssl.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnssl.paths import nnssl_raw, nnssl_preprocessed
from batchgenerators.utilities.file_and_folder_operations import load_json, join, save_json, isfile, maybe_mkdir_p
from nnssl.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnssl.data.utils import get_train_dataset
from nnssl.experiment_planning.dataset_fingerprint.utils import analyze_case


def setup_dataset_fingerprint_extractor(
    dataset_name_or_id: Union[str, int], num_processes: int = 8, verbose: bool = False
):
    """
    Sets up the dataset fingerprint extractor.

    Args:
        dataset_name_or_id (Union[str, int]): The name or ID of the dataset.
        num_processes (int, optional): The number of processes to use for extraction. Defaults to 8.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        tuple: A tuple containing the dataset name, number of processes, dataset JSON, and dataset.
    """

    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    input_folder = join(nnssl_raw, dataset_name)
    dataset_json = load_json(join(input_folder, "dataset.json"))

    dataset = get_train_dataset(input_folder, dataset_json)
    return dataset_name, num_processes, dataset_json, dataset


def save_fingerprint(fingerprint, properties_file):
    """
    Save the fingerprint to a JSON file.

    Args:
        fingerprint (dict): The fingerprint to be saved.
        properties_file (str): The path to the JSON file.

    Raises:
        Exception: If there is an error saving the fingerprint.
    """
    try:
        save_json(fingerprint, properties_file)
    except Exception as e:
        if isfile(properties_file):
            os.remove(properties_file)
        raise e


def default_dataset_fingerprint_extraction(
    dataset_name_or_id: Union[str, int],
    num_processes: int = 8,
    verbose: bool = False,
    overwrite_existing: bool = False,
) -> dict:
    """
    Runs the dataset fingerprint extraction process.

    Args:
        dataset_name_or_id (Union[str, int]): The name or ID of the dataset.
        num_processes (int, optional): The number of processes to use for parallel execution. Defaults to 8.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        overwrite_existing (bool, optional): Whether to overwrite existing fingerprint data. Defaults to False.

    Returns:
        dict: The dataset fingerprint containing spacings, shapes after crop, and median relative size after cropping.
    """

    (
        dataset_name,
        num_processes,
        dataset_json,
        dataset,
    ) = setup_dataset_fingerprint_extractor(dataset_name_or_id, num_processes, verbose)

    preprocessed_output_folder = join(nnssl_preprocessed, dataset_name)
    maybe_mkdir_p(preprocessed_output_folder)
    properties_file = join(preprocessed_output_folder, "dataset_fingerprint.json")

    if not isfile(properties_file) or overwrite_existing:
        reader_writer_class = determine_reader_writer_from_dataset_json(dataset_json, dataset.get_all_image_paths()[0])
        analyze_case_partial = partial(analyze_case, reader_writer_class=reader_writer_class)
        if num_processes > 1:
            with multiprocessing.get_context("spawn").Pool(num_processes) as p:
                results = list(tqdm(p.imap(analyze_case_partial, [[k] for k in dataset.get_all_image_paths()])))
        else:
            results = [analyze_case([k], reader_writer_class) for k in tqdm(dataset)]

        shapes_after_crop = [r[0] for r in results]
        spacings = [r[1] for r in results]
        median_relative_size_after_cropping = np.median([r[2] for r in results], 0)

        fingerprint = {
            "spacings": spacings,
            "shapes_after_crop": shapes_after_crop,
            "median_relative_size_after_cropping": median_relative_size_after_cropping,
        }

        save_fingerprint(fingerprint, properties_file)
    else:
        fingerprint = load_json(properties_file)

    return fingerprint


if __name__ == "__main__":
    fingerprint = default_dataset_fingerprint_extraction(2, 8, verbose=False, overwrite_existing=False)
