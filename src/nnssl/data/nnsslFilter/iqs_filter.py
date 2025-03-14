from nnssl.data.raw_dataset import IndependentImage, Collection
from nnssl.data.nnsslFilter.abstract_filter import AbstractFilter


class OpenMindIQSFilter(AbstractFilter):
    """
    Image Quality Score (IQS) Filter. Excludes images that have an IQS greater than the threshold.
    """

    def __init__(self, collection: Collection, threshold: float):
        self.threshold = threshold # change this in child classes
        self.iqs_dict: dict[frozenset, float] = {}

        for dataset in collection.datasets.values():
            dicts = dataset.dataset_info["image_quality_score"]
            for dic in dicts:
                iqs_value = dic.pop("image_quality_score")
                dic["dataset_id"] = dataset.dataset_index
                self.iqs_dict[frozenset(dic.items())] = iqs_value

    def __call__(self, iimg: IndependentImage) -> bool:
        dataset_id = iimg.dataset_index
        modality = iimg.image_modality
        derived_from = iimg.image_info.get("derived_from")

        key = frozenset({
            "dataset_id": dataset_id,
            "modality": modality,
            "derived_from": derived_from
        }.items())
        return self.iqs_dict[key] <= self.threshold
