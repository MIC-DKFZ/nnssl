import torch
from nnssl.experiment_planning.experiment_planners.plan import Plan
from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import BaseMAETrainer
import numpy as np


class BaseMAETrainer_BS16_ep1000(BaseMAETrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 16
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.num_epochs = 1000


class BaseMAETrainer_BS24_ep1000(BaseMAETrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 24
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.num_epochs = 1000


class BaseMAETrainer_BS32_ep1000(BaseMAETrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 32
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.num_epochs = 1000
