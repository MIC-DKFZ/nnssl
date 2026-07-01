from typing import List, Literal, Union, Tuple

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json, join, save_json, isfile, maybe_mkdir_p

from nnssl.data.raw_dataset import Collection
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.imageio.reader_writer_registry import (
    determine_reader_writer_from_file_ending,
)
from nnssl.paths import nnssl_raw, nnssl_preprocessed
from nnssl.preprocessing.preprocessors.abstract_preprocessor import Preprocessors
from nnssl.preprocessing.preprocessors.default_preprocessor import PREPROCESS_SPACING_STYLES
from nnssl.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
from nnssl.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnssl.utilities.json_export import recursive_fix_for_json_export
from nnssl.data.utils import get_train_collection

data_spacing_config = Literal["median", "onemmiso", "noresample"]


class ExperimentPlanner(object):
    def __init__(
        self,
        dataset_name_or_id: Union[str, int],
        plans_name: str = "nnsslPlans",
        suppress_transpose: bool = False,
    ):
        """
        overwrite_target_spacing only affects 3d_fullres! (but by extension 3d_lowres which starts with fullres may
        also be affected
        """

        self.dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
        self.suppress_transpose = suppress_transpose
        self.raw_dataset_folder = join(nnssl_raw, self.dataset_name)
        preprocessed_folder = join(nnssl_preprocessed, self.dataset_name)

        self.collection: Collection = get_train_collection(self.raw_dataset_folder)

        # load dataset fingerprint
        if not isfile(join(preprocessed_folder, "dataset_fingerprint.json")):
            raise RuntimeError("Fingerprint missing for this dataset. Please run nnUNet_extract_dataset_fingerprint")

        self.dataset_fingerprint = load_json(join(preprocessed_folder, "dataset_fingerprint.json"))
        self.plans_identifier = plans_name

        self.plans = None

    def determine_reader_writer(self):
        example_image = self.collection.get_all_image_paths()[0]
        if example_image.endswith(".gz"):
            file_extension = "." + ".".join(str(example_image).split(".")[-2:])
        else:
            file_extension = "." + example_image.split(".")[-1]
        # Extensions always look like:  ".nii.gz"
        return determine_reader_writer_from_file_ending(file_extension, example_image)

    def determine_resampling(self, *args, **kwargs):
        """
        returns what functions to use for resampling data and seg, respectively. Also returns kwargs
        resampling function must be callable(data, current_spacing, new_spacing, **kwargs)

        determine_resampling is called within get_plans_for_configuration to allow for different functions for each
        configuration
        """
        resampling_data = resample_data_or_seg_to_shape
        resampling_data_kwargs = {
            "is_seg": False,
            "order": 3,
            "order_z": 0,
            "force_separate_z": None,
        }
        resampling_seg = resample_data_or_seg_to_shape
        resampling_seg_kwargs = {
            "is_seg": True,
            "order": 1,
            "order_z": 0,
            "force_separate_z": None,
        }
        return resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs

    def determine_segmentation_softmax_export_fn(self, *args, **kwargs):
        """
        function must be callable(data, new_shape, current_spacing, new_spacing, **kwargs). The new_shape should be
        used as target. current_spacing and new_spacing are merely there in case we want to use it somehow

        determine_segmentation_softmax_export_fn is called within get_plans_for_configuration to allow for different
        functions for each configuration

        """
        resampling_fn = resample_data_or_seg_to_shape
        resampling_fn_kwargs = {
            "is_seg": False,
            "order": 1,
            "order_z": 0,
            "force_separate_z": None,
        }
        return resampling_fn, resampling_fn_kwargs

    def determine_fullres_target_spacing(self) -> np.ndarray:
        """
        per default we use the 50th percentile=median for the target spacing. Higher spacing results in smaller data
        and thus faster and easier training. Smaller spacing results in larger data and thus longer and harder training

        For some datasets the median is not a good choice. Those are the datasets where the spacing is very anisotropic
        (for example ACDC with (10, 1.5, 1.5)). These datasets still have examples with a spacing of 5 or 6 mm in the low
        resolution axis. Choosing the median here will result in bad interpolation artifacts that can substantially
        impact performance (due to the low number of slices).
        """
        spacings = self.dataset_fingerprint["spacings"]
        # ToDo: Add some k-means clustering approach here just because I am curious.
        #   ToDo: Conduct this for Fabi's large 120-ish Dataset collection (just the downstream medical task?)
        target = np.percentile(np.vstack(spacings), 50, 0)

        return target

    def determine_transpose(self):
        if self.suppress_transpose:
            return [0, 1, 2], [0, 1, 2]

        target_spacing = self.determine_fullres_target_spacing()

        max_spacing_axis = np.argmax(target_spacing)
        remaining_axes = [i for i in list(range(3)) if i != max_spacing_axis]
        transpose_forward = [max_spacing_axis] + remaining_axes
        transpose_backward = [np.argwhere(np.array(transpose_forward) == i)[0][0] for i in range(3)]
        return transpose_forward, transpose_backward

    def get_plans_for_configuration(
        self,
        config: data_spacing_config,
        spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
        data_identifier: str,
    ) -> ConfigurationPlan:
        assert all([i > 0 for i in spacing]), f"Spacing must be > 0! Spacing: {spacing}"
        resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs = self.determine_resampling()

        spacing_style: PREPROCESS_SPACING_STYLES
        if config == "median":
            spacing_style = "median"
        elif config == "onemmiso":
            spacing = [1, 1, 1]
            spacing_style = "onemmiso"
        elif config == "noresample":
            spacing = None
            spacing_style = "noresample"
        else:
            raise NotImplementedError()

        plan = {
            "data_identifier": data_identifier,
            "preprocessor_name": Preprocessors.DEFAULT.value,
            "spacing_style": spacing_style,
            "spacing": spacing,
            "normalization_schemes": ["ZScoreNormalization"],
            "use_mask_for_norm": [False],
            "resampling_fn_data": resampling_data.__name__,
            "resampling_fn_data_kwargs": resampling_data_kwargs,
            "resampling_fn_mask": resampling_seg.__name__,
            "resampling_fn_mask_kwargs": resampling_seg_kwargs,
        }

        return ConfigurationPlan(**plan)

    def plan_experiment(self) -> Plan:
        """
        MOVE EVERYTHING INTO THE PLANS. MAXIMUM FLEXIBILITY

        Ideally I would like to move transpose_forward/backward into the configurations so that this can also be done
        differently for each configuration but this would cause problems with identifying the correct axes for 2d. There
        surely is a way around that but eh. I'm feeling lazy and featuritis must also not be pushed to the extremes.

        So for now if you want a different transpose_forward/backward you need to create a new planner. Also not too
        hard.
        """
        # first get transpose
        transpose_forward, transpose_backward = self.determine_transpose()

        # get fullres spacing and transpose it
        fullres_spacing = self.determine_fullres_target_spacing()
        fullres_spacing_transposed = fullres_spacing[transpose_forward]

        # ----------- We always plan all three because it's cheap and easy ----------- #
        median = self.get_plans_for_configuration(
            "median",
            fullres_spacing_transposed,
            self.generate_data_identifier("median"),
        )
        onemmiso = self.get_plans_for_configuration(
            "onemmiso",
            fullres_spacing_transposed,
            self.generate_data_identifier("onemmiso"),
        )
        noresample = self.get_plans_for_configuration(
            "noresample",
            fullres_spacing_transposed,
            self.generate_data_identifier("noresample"),
        )

        median_spacing = np.median(self.dataset_fingerprint["spacings"], 0)[transpose_forward]
        # json is stupid and I hate it... "Object of type int64 is not JSON serializable" -> my ass
        plans = Plan(
            **{
                "dataset_name": self.dataset_name,
                "plans_name": self.plans_identifier,
                "original_median_spacing_after_transp": [float(i) for i in median_spacing],
                "image_reader_writer": self.determine_reader_writer().__name__,
                "transpose_forward": [int(i) for i in transpose_forward],
                "transpose_backward": [int(i) for i in transpose_backward],
                "configurations": {},
                "experiment_planner_used": self.__class__.__name__,
            }
        )

        plans["configurations"]["median"] = median
        plans["configurations"]["onemmiso"] = onemmiso
        plans["configurations"]["noresample"] = noresample

        self.plans: Plan = plans
        plans.save_to_file(overwrite=True)
        return plans

    def save_plans(self, plans: Plan):
        recursive_fix_for_json_export(plans)

        plans_file = join(nnssl_preprocessed, self.dataset_name, self.plans_identifier + ".json")

        # we don't want to overwrite potentially existing custom configurations every time this is executed. So let's
        # read the plans file if it already exists and keep any non-default configurations
        if isfile(plans_file):
            old_plans = load_json(plans_file)
            old_configurations = old_plans["configurations"]
            for c in plans["configurations"].keys():
                if c in old_configurations.keys():
                    del old_configurations[c]
            plans["configurations"].update(old_configurations)

        maybe_mkdir_p(join(nnssl_preprocessed, self.dataset_name))
        save_json(plans, plans_file, sort_keys=False)
        print(f"Plans were saved to {join(nnssl_preprocessed, self.dataset_name, self.plans_identifier + '.json')}")

    def generate_data_identifier(self, configuration_name: str) -> str:
        """
        configurations are unique within each plans file but different plans file can have configurations with the
        same name. In order to distinguish the associated data we need a data identifier that reflects not just the
        config but also the plans it originates from
        """
        return self.plans_identifier + "_" + configuration_name

    def load_plans(self, fname: str) -> Plan:
        self.plans = Plan.load_from_file(fname)


if __name__ == "__main__":
    ExperimentPlanner(2, 8).plan_experiment()
