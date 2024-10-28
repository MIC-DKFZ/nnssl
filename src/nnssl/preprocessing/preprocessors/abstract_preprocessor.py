import argparse
from enum import Enum
from typing import Protocol, Union

from nnssl.preprocessing.preprocessors.default_preprocessor import default_preprocess


class PreprocessorProtocol(Protocol):
    """Preprocessor protocol."""

    def __call__(
        dataset_name_or_id: Union[int, str],
        configuration_name: str,
        plans_identifier: str,
        part: int,
        total_parts: int,
        num_processes: int,
        verbose: bool = True,
    ) -> None: ...


class Preprocessors(Enum):
    DEFAULT = "DefaultPreprocessor"


def preprocessor_type(value):
    try:
        return Preprocessors(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid option")


def get_preprocessor(preprocessor: Preprocessors) -> PreprocessorProtocol:
    """
    Returns the appropriate preprocessor based on the given preprocessor type.

    Args:
        preprocessor (Preprocessors): The type of preprocessor.

    Returns:
        PreprocessorProtocol: The preprocessor.

    Raises:
        ValueError: If the given preprocessor type is unknown.
    """
    if preprocessor == Preprocessors.DEFAULT.value:
        return default_preprocess
    else:
        raise ValueError(f"Unknown preprocessor: {preprocessor}")
