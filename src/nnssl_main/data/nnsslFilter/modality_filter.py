from nnssl.data.raw_dataset import IndependentImage
from nnssl.data.nnsslFilter.abstract_filter import AbstractFilter


class ModalityFilter(AbstractFilter):
    """
    Modality Filter. Excludes image if its modality is not in valid_modalities.
    """

    def __init__(self, valid_modalities: list[str]):
        self.valid_modalites = set(valid_modalities)

    def __call__(self, iimg: IndependentImage) -> bool:
        return iimg.image_modality in self.valid_modalites
