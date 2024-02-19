from torch import nn
import torch
from nnssl.training.loss.abstract_loss import AbstractLoss
from einops import repeat


class SparkLoss(AbstractLoss):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction="none")

    def forward(self, prediction: torch.Tensor, groundtruth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Can take any outputs,  ."""

        out_D, out_H, out_W = prediction.shape[2], prediction.shape[3], prediction.shape[4]
        D, H, W = mask.shape[2], mask.shape[3], mask.shape[4]
        d_repeat, h_repeat, w_repeat = out_D // D, out_H // H, out_W // W
        loss_mask = (
            mask.repeat_interleave(d_repeat, dim=2)
            .repeat_interleave(h_repeat, dim=3)
            .repeat_interleave(w_repeat, dim=4)
        )
        loss_mask = 1 - loss_mask  # We want to only penalize where the mask is NOT active!
        # Mask = 1 represents not masked points
        diff = (groundtruth - prediction) ** 2  # (B, 1, D, H, W) (same as mask (B, 1, D, H, W))
        reconstruction_loss = torch.mean(diff[loss_mask.nonzero(as_tuple=True)])
        return reconstruction_loss
