import torch
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgenerators.transforms.utility_transforms import RenameTransform


class SimCLRTransform(AbstractTransform):
    def __init__(self, transforms):
        """
        The SimCLR Transform takes the regular transforms and applies them to the same data twice.

        return tuple of transformed data_dicts
        """
        self.transforms = transforms
        self.rename = RenameTransform(in_key="data", out_key="image", delete_old=True)

    def __call__(self, **data_dict):
        renamed = self.rename(**data_dict)
        renamed["image"] = torch.from_numpy(renamed["image"]).squeeze().float()
        xi = self.transforms(**renamed)
        xj = self.transforms(**renamed)

        return {"image_i": xi["image"], "image_j": xj["image"]}
