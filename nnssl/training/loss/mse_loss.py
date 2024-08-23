from torch import nn
import torch
from nnssl.training.loss.abstract_loss import AbstractLoss


class MAEMSELoss(AbstractLoss):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction="none")

    def forward(
        self, model_output: torch.Tensor, target: dict[str, torch.Tensor], mask: torch.Tensor
    ) -> torch.Tensor:
        """Can take any outputs,  ."""
        # Mask = 1 represents not masked points
        reconstruction_loss = (model_output - target) ** 2  # (B, X, Y, Z, C)
        reconstruction_loss = torch.sum(reconstruction_loss * (1 - mask)) / torch.sum((1 - mask))

        return reconstruction_loss
