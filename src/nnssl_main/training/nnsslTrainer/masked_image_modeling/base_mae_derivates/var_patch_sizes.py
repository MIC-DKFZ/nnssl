import torch
from nnssl.experiment_planning.experiment_planners.plan import Plan
from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import BaseMAETrainer, BaseMAETrainer_BS8_1000ep
import numpy as np


class BaseMAETrainer_BS8_ep1000_patch_128(BaseMAETrainer_BS8_1000ep):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].patch_size = [128, 128, 128]
        super().__init__(plan, configuration_name, fold, pretrain_json, device)


class BaseMAETrainer_BS8_ep1000_patch_160(BaseMAETrainer_BS8_1000ep):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].patch_size = [160, 160, 160]
        super().__init__(plan, configuration_name, fold, pretrain_json, device)


class BaseMAETrainer_BS8_ep1000_patch_224(BaseMAETrainer_BS8_1000ep):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].patch_size = [224, 224, 224]
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
