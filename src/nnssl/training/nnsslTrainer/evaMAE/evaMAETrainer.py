import torch
from torch import nn

from nnssl.training.nnsslTrainer.evaMAE.evaMAE_module import EvaMAE

from nnssl.experiment_planning.experiment_planners.plan import Plan
from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer
from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import BaseMAETrainer
import numpy as np

class EvaMAETrainer(BaseMAETrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.mask_ratio = 0.5

    def build_architecture(self, *args, **kwargs) -> nn.Module:
        network = EvaMAE(
            input_channels=1,
            embed_dim=192,
            patch_embed_size=(8, 8, 8),
            output_channels=3,
            input_shape=self.config_plan.patch_size,
            eva_depth=6,
            eva_numheads=8,
            patch_drop_rate=self.mask_ratio
        )
        return network