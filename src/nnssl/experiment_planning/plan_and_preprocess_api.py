from typing import List, Sequence, Type, Tuple, Union, TYPE_CHECKING

import nnssl
from batchgenerators.utilities.file_and_folder_operations import join

from nnssl.experiment_planning.dataset_fingerprint.default_fingerprint_extractor import (
    default_dataset_fingerprint_extraction,
)
from nnssl.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from nnssl.experiment_planning.experiment_planners.plan import Plan
from nnssl.paths import nnssl_preprocessed
from nnssl.preprocessing.preprocessors.abstract_preprocessor import get_preprocessor

if TYPE_CHECKING:
    from nnssl.preprocessing.preprocessors.abstract_preprocessor import PreprocessorProtocol
from nnssl.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
from nnssl.utilities.find_class_by_name import recursive_find_python_class
from nnssl.configuration import default_num_processes


def extract_fingerprint_dataset(
    dataset_id: int,
    num_processes: int = default_num_processes,
    clean: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Returns the fingerprint as a dictionary (additionally to saving it)
    """
    dataset_name = convert_id_to_dataset_name(dataset_id)
    print(dataset_name)

    fingerprint = default_dataset_fingerprint_extraction(
        dataset_id, num_processes, verbose=verbose, overwrite_existing=clean
    )
    return fingerprint


def extract_fingerprints(
    dataset_ids: List[int],
    num_processes: int = default_num_processes,
    clean: bool = True,
    verbose: bool = True,
):
    """
    Extracts fingerprints from the specified dataset IDs using the given fingerprint extractor.

    Args:
        dataset_ids (List[int]): The list of dataset IDs to extract fingerprints from.
        fingerprint_extractor_class_name (DatasetFingerprintExtractor): The (enum) class name of the fingerprint extractor to use.
        num_processes (int, optional): The number of processes to use for parallel extraction. Defaults to default_num_processes.
        check_dataset_integrity (bool, optional): Whether to check the integrity of the dataset before extraction. Defaults to False.
        clean (bool, optional): Whether to clean the extracted fingerprints. Defaults to True.
        verbose (bool, optional): Whether to print verbose output during extraction. Defaults to True.
    """
    for d in dataset_ids:
        extract_fingerprint_dataset(d, num_processes, clean, verbose)


def plan_experiment_dataset(
    dataset_id: int,
    experiment_planner_class: Type[ExperimentPlanner] = ExperimentPlanner,
) -> Plan:
    """
    overwrite_target_spacing ONLY applies to 3d_fullres and 3d_cascade fullres!
    """
    kwargs = {}
    return experiment_planner_class(
        dataset_id,
        suppress_transpose=False,  # might expose this later,
        **kwargs,
    ).plan_experiment()


def plan_experiments(
    dataset_ids: List[int],
    experiment_planner_class_name: str = "ExperimentPlanner",
):
    """
    overwrite_target_spacing ONLY applies to 3d_fullres and 3d_cascade fullres!
    """
    experiment_planner = recursive_find_python_class(
        join(nnssl.__path__[0], "experiment_planning"),
        experiment_planner_class_name,
        current_module="nnssl.experiment_planning",
    )
    plans = []
    for d in dataset_ids:
        plans.append(
            plan_experiment_dataset(
                d,
                experiment_planner,
            )
        )
    return plans


def preprocess_dataset(
    dataset_id: int,
    plans_identifier: str = "nnsslPlans",
    configurations: Union[Tuple[str], List[str]] = ("onemmiso",),
    part: int = 0,
    total_parts: int = 1,
    num_processes: Sequence[int] = (4,),
    verbose: bool = False,
) -> None:
    if not isinstance(num_processes, list):
        num_processes = list(num_processes)
    if len(num_processes) == 1:
        num_processes = num_processes * len(configurations)
    if len(num_processes) != len(configurations):
        raise RuntimeError(
            f"The list provided with num_processes must either have len 1 or as many elements as there are "
            f"configurations (see --help). Number of configurations: {len(configurations)}, length "
            f"of num_processes: "
            f"{len(num_processes)}"
        )

    dataset_name = convert_id_to_dataset_name(dataset_id)
    print(f"Preprocessing dataset {dataset_name}")
    plans_file = join(nnssl_preprocessed, dataset_name, plans_identifier + ".json")
    plan: Plan = Plan.load_from_file(plans_file)
    for n, c in zip(num_processes, configurations):
        print(f"Configuration: {c}...")
        if c not in plan.configurations.keys():
            print(
                f"INFO: Configuration {c} not found in plans file {plans_identifier + '.json'} of "
                f"dataset {dataset_name}. Skipping."
            )
            continue
        config = plan.configurations[c]
        preprocessor: PreprocessorProtocol = get_preprocessor(config.preprocessor_name)
        preprocessor(dataset_id, c, plans_identifier, part, total_parts, num_processes=n)


def preprocess(
    dataset_ids: List[int],
    plans_identifier: str = "nnsslPlans",
    part: int = 0,
    total_parts: int = 1,
    configurations: Union[Tuple[str], List[str]] = ("onemmiso",),
    num_processes: Union[int, Tuple[int, ...], List[int]] = (4),
    verbose: bool = False,
):
    for d in dataset_ids:
        preprocess_dataset(d, plans_identifier, configurations, part, total_parts, num_processes, verbose)
