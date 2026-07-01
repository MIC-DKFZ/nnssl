from typing import List, Tuple, Union
import torch
from nnssl.experiment_planning.experiment_planners.plan import Plan
from nnssl.ssl_data.data_augmentation.transforms_for_dummy_2d import Convert2DTo3DTransform, Convert3DTo2DTransform
from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import BaseMAETrainer_BS8_1000ep
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor
import numpy as np


class BaseMAETrainer_BS8_ep2000(BaseMAETrainer_BS8_1000ep):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        # plan.configurations[configuration_name].batch_size = 1
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.num_epochs = 2000


class BaseMAETrainer_BS8_ep3000(BaseMAETrainer_BS8_1000ep):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        # plan.configurations[configuration_name].batch_size = 1
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.num_epochs = 3000


class BaseMAETrainer_BS8_ep4000(BaseMAETrainer_BS8_1000ep):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        # plan.configurations[configuration_name].batch_size = 1
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.num_epochs = 4000
