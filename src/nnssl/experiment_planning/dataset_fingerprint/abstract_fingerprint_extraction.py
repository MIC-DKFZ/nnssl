import argparse
from enum import Enum
from typing import Union, Dict, Protocol

from nnssl.experiment_planning.dataset_fingerprint.default_fingerprint_extractor import (
    default_dataset_fingerprint_extraction,
)


class FingerprintExtractionProtocol(Protocol):
    """
    Protocol for fingerprint extraction.

    Args:
        dataset_name_or_id (Union[str, int]): The name or ID of the dataset.
        num_processes (int): The number of processes to use for extraction.
        verbose (bool): Whether to display verbose output.
        overwrite_existing (bool): Whether to overwrite existing fingerprints.

    Returns:
        Dict: A dictionary containing the extracted fingerprints.
    """

    def __call__(
        self, dataset_name_or_id: Union[str, int], num_processes: int, verbose: bool, overwrite_existing: bool
    ) -> Dict:
        ...


class DatasetFingerprintExtractor(Enum):
    DEFAULT = "DatasetFingerprintExtractor"


def fingerprint_type(value):
    try:
        return DatasetFingerprintExtractor(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid option")


def get_dataset_fingerprint_extractor(extractor: DatasetFingerprintExtractor) -> FingerprintExtractionProtocol:
    """
    Returns the appropriate dataset fingerprint extractor based on the given extractor type.

    Args:
        extractor (DatasetFingerprintExtractor): The type of dataset fingerprint extractor.

    Returns:
        FingerprintExtractionProtocol: The dataset fingerprint extractor.

    Raises:
        ValueError: If the given extractor type is unknown.
    """
    if extractor == DatasetFingerprintExtractor.DEFAULT:
        return default_dataset_fingerprint_extraction
    else:
        raise ValueError(f"Unknown dataset fingerprint extractor: {extractor}")
