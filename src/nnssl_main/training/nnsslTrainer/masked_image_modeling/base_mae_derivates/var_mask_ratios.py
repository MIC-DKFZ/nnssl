import os
import torch
from nnssl.experiment_planning.experiment_planners.plan import Plan
from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import BaseMAETrainer_BS8_1000ep


class BaseMAETrainer_BS8_ep1000_mask30(BaseMAETrainer_BS8_1000ep):

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
        self.mask_percentage: float = 0.30


class BaseMAETrainer_BS8_ep1000_mask45(BaseMAETrainer_BS8_1000ep):

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
        self.mask_percentage: float = 0.45


class BaseMAETrainer_BS8_ep1000_mask60(BaseMAETrainer_BS8_1000ep):

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
        self.mask_percentage: float = 0.60



class BaseMAETrainer_BS8_ep1000_mask90(BaseMAETrainer_BS8_1000ep):

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
        self.mask_percentage: float = 0.90
