from nnssl.data.raw_dataset import IndependentImage


class AbstractFilter:
    """
    Base class for all filters that select images of a Collection to be part of the Dataset.
    """

    def __call__(
        self, iimg: IndependentImage
    ) -> bool:
        """
        Args:
            iimg (IndependentImage): The image in question.

        Returns:
            bool: Returns true if the image should be included during training, false if otherwise.
        """

        ...