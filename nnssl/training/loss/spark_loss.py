from torch import nn
import torch
from nnssl.training.loss.abstract_loss import AbstractLoss


class SparkLoss(AbstractLoss):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction="none")

    def forward(self, model_output: torch.Tensor, target: dict[str, torch.Tensor]) -> torch.Tensor:
        """Can take any outputs,  ."""
        reconstruction, mask = target["prediction"], target["mask"]
        non_active = torch.logical_not(mask).float()
        # Mask = 1 represents not masked points
        diff = torch.mean((model_output - reconstruction) ** 2, dim=-1, keepdim=True)  # (B, X, Y, Z, C)
        masked_diff = diff * non_active
        reconstruction_loss = masked_diff.sum() / (non_active + 1e-8)
        return reconstruction_loss
