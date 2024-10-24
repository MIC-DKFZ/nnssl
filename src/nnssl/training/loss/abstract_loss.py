from torch import nn
from abc import ABC, abstractmethod

import torch


class AbstractLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, model_output: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> torch.Tensor:
        """Can take any outputs,  ."""
        pass
