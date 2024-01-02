from torch import nn
import torch
from nnssl.training.loss.abstract_loss import AbstractLoss


class MSELoss(AbstractLoss):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction="none")

    def forward(self, model_output: torch.Tensor, target: dict[str, torch.Tensor]) -> torch.Tensor:
        """Can take any outputs,  ."""
        reconstruction = target["target"]
        # Mask = 1 represents not masked points
        reconstruction_loss = torch.mean((model_output - reconstruction) ** 2)  # (B, X, Y, Z, C)
        return reconstruction_loss
