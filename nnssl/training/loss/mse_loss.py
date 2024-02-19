from torch import nn
import torch
from nnssl.training.loss.abstract_loss import AbstractLoss


class MSELoss(AbstractLoss):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction="none")

    def forward(self, prediction: torch.Tensor, groundtruth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Can take any outputs,  ."""

        # Mask = 1 represents not masked points
        reconstruction_loss = torch.mean((prediction - groundtruth) ** 2)  # (B, X, Y, Z, C)
        return reconstruction_loss


class MAEMSELoss(AbstractLoss):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction="none")

    def forward(
        self, model_output: torch.Tensor, target: dict[str, torch.Tensor], mask: torch.Tensor
    ) -> torch.Tensor:
        """Can take any outputs,  ."""
        reconstruction = target["target"]
        mask = target["mask"]
        # Mask = 1 represents not masked points
        reconstruction_loss = (model_output - reconstruction) ** 2  # (B, X, Y, Z, C)
        reconstruction_loss = torch.sum(reconstruction_loss * mask) / torch.sum(mask)

        return reconstruction_loss
