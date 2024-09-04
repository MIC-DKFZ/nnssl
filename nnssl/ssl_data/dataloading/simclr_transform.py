from batchgenerators.transforms.abstract_transforms import AbstractTransform


class SimCLRTransform(AbstractTransform):
    def __init__(self, transforms):
        """
        The SimCLR Transform takes the regular transforms and applies them to the same data twice.

        return tuple of transformed data_dicts
        """
        self.transforms = transforms

    def __call__(self, **data_dict):
        xi = self.transforms(**data_dict)
        xj = self.transforms(**data_dict)

        return xi, xj
