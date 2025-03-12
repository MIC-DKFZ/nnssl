from copy import deepcopy
from typing import List, Tuple, Union

import numpy as np
from nnssl.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner

from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan


class OneMMPlanner(ExperimentPlanner):

    def __init__(
        self,
        dataset_name_or_id: Union[str, int],
        preprocessor_name: str = "DefaultPreprocessor",
        plans_name: str = "onemmPlans",
        suppress_transpose: bool = False,
    ):
        super().__init__(
            dataset_name_or_id,
            preprocessor_name,
            plans_name,
            suppress_transpose,
        )

    def determine_fullres_target_spacing(self) -> np.ndarray:
        """
        per default we use the 50th percentile=median for the target spacing. Higher spacing results in smaller data
        and thus faster and easier training. Smaller spacing results in larger data and thus longer and harder training

        For some datasets the median is not a good choice. Those are the datasets where the spacing is very anisotropic
        (for example ACDC with (10, 1.5, 1.5)). These datasets still have examples with a spacing of 5 or 6 mm in the low
        resolution axis. Choosing the median here will result in bad interpolation artifacts that can substantially
        impact performance (due to the low number of slices).
        """
        target = [1.0, 1.0, 1.0]
        return target

    def get_plans_for_configuration(
        self,
        spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
        data_identifier: str,
    ) -> ConfigurationPlan:
        assert all([i > 0 for i in spacing]), f"Spacing must be > 0! Spacing: {spacing}"

        # use that to get the network topology. Note that this changes the patch_size depending on the number of
        # pooling operations (must be divisible by 2**num_pool in each axis)

        resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs = self.determine_resampling()

        plan = {
            "data_identifier": data_identifier,
            "preprocessor_name": self.preprocessor_name,
            "spacing": spacing,
            "normalization_schemes": "ZScoreNormalization",
            "use_mask_for_norm": [False],
            "resampling_fn_data": resampling_data.__name__,
            "resampling_fn_data_kwargs": resampling_data_kwargs,
            "resampling_fn_mask": resampling_seg.__name__,
            "resampling_fn_mask_kwargs": resampling_seg_kwargs,
            "batch_dice": False,
        }

        return ConfigurationPlan(**plan)
