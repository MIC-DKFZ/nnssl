from random import sample
import shutil
from tabnanny import verbose

import numpy as np
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.data.utils import get_train_dataset
from batchgenerators.utilities.file_and_folder_operations import *
from nnssl.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
import SimpleITK as sitk

nnssl_preprocessed = os.environ["nnssl_preprocessed"]


def convert_npzs_to_nifti_im(npz_file: str, config_plan: ConfigurationPlan) -> sitk.Image:
    spacing = config_plan.spacing
    data = np.load(npz_file, "r")["data"][0]
    im = sitk.GetImageFromArray(data)
    im.SetSpacing(spacing)
    return im


def create_pp_niftis(dataset_name_or_id: int, plans_identifier: str, configuration_name: str, n_samples: int = 20):
    """
    Main function that is called externally.
    Does the preprocessing of the cases found in the dataset_name.
    This is the nnssl version, where we neglect any labels that may be present and create a new dataset.json
    that does not contain label information.
    """
    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)

    plans_file = join(nnssl_preprocessed, dataset_name, plans_identifier + ".json")
    assert isfile(plans_file), (
        "Expected plans file (%s) not found. Run corresponding nnUNet_plan_experiment " "first." % plans_file
    )
    plan: Plan = Plan.load_from_file(plans_file)
    config_plan: ConfigurationPlan = plan.configurations[configuration_name]

    if verbose:
        print(f"Preprocessing the following configuration: {configuration_name}")
        print(config_plan)

    dataset_json_file = join(nnssl_preprocessed, dataset_name, "dataset.json")
    dataset_json = load_json(dataset_json_file)

    output_directory = join(nnssl_preprocessed, dataset_name, (config_plan.data_identifier + "_pp_nifti_samples"))
    if isdir(output_directory):
        shutil.rmtree(output_directory)  # remove the folder if it exists
    maybe_mkdir_p(output_directory)

    data_dir = join(nnssl_preprocessed, dataset_name, config_plan.data_identifier)
    npzs = [join(data_dir, i) for i in os.listdir(data_dir) if i.endswith(".npz")]
    npz_samples = sample(npzs, n_samples)

    for npz in npz_samples:
        im = convert_npzs_to_nifti_im(npz, config_plan)
        sitk.WriteImage(im, join(output_directory, os.path.basename(npz)[:-4] + ".nii.gz"))

    # identifiers = [os.path.basename(i[:-len(dataset_json['file_ending'])]) for i in seg_fnames]
    # output_filenames_truncated = [join(output_directory, i) for i in identifiers]


if __name__ == "__main__":
    dataset_id = 741
    plans_id = "nnsslPlans"
    config_name = "3d_fullres"
    create_pp_niftis(dataset_id, plans_id, config_name, n_samples=20)
