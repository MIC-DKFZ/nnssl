from torch import nn
import torch
from nnssl.training.loss.abstract_loss import AbstractLoss
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from einops import rearrange


class MAEMSELoss(AbstractLoss):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction="none")

    def forward(
        self, model_output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Can take any outputs"""
        # Mask = 1 represents not masked points
        reconstruction_loss = (model_output - target) ** 2  # (B, X, Y, Z, C)
        reconstruction_loss = torch.sum(reconstruction_loss * (1 - mask)) / torch.sum((1 - mask))

        return reconstruction_loss

class LossMaskMSELoss(AbstractLoss):
    def forward(
        self, model_output: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor
    ) -> torch.Tensor:
        """loss_mask = 1 in positions where loss calculation should take place"""
        reconstruction_loss = (model_output - target) ** 2  # (B, X, Y, Z, C)
        reconstruction_loss = torch.sum(reconstruction_loss * loss_mask) / torch.sum(loss_mask)

        return reconstruction_loss


class MAEL1Loss(AbstractLoss):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss(reduction="none")

    def forward(
        self, model_output: torch.Tensor, target: dict[str, torch.Tensor], mask: torch.Tensor
    ) -> torch.Tensor:
        """Can take any outputs,  ."""
        # Mask = 1 represents not masked points
        reconstruction_loss = torch.sum(torch.abs(model_output - target) * (1 - mask)) / torch.sum((1 - mask))

        return reconstruction_loss


class MAESSIMLoss(AbstractLoss):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction="none")

    def forward(
        self, model_output: torch.Tensor, target: dict[str, torch.Tensor], mask: torch.Tensor
    ) -> torch.Tensor:
        """Can take any outputs,  ."""
        # Mask = 1 represents not masked points
        # Rescale output and target to [0, 1] for each image in batch
        output_min = torch.amin(model_output, dim=(1, 2, 3, 4), keepdim=True)
        output_max = torch.amax(model_output, dim=(1, 2, 3, 4), keepdim=True)
        target_min = torch.amin(target, dim=(1, 2, 3, 4), keepdim=True)
        target_max = torch.amax(target, dim=(1, 2, 3, 4), keepdim=True)

        rescaled_out = (model_output - output_min) / (output_max - output_min)
        rescaled_target = (target - target_min) / (target_max - target_min)

        # rescaled_out = rearrange(rescaled_out, "b x y z c -> b c x y z")
        # rescaled_target = rearrange(rescaled_target, "b x y z c -> b c x y z")

        ssim_loss = 1 - ssim(rescaled_out, rescaled_target, data_range=1, size_average=False, nonnegative_ssim=True)

        return torch.mean(ssim_loss)


class MAESSIMLoss_WithMask(AbstractLoss):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction="none")

    def forward(
        self, model_output: torch.Tensor, target: dict[str, torch.Tensor], mask: torch.Tensor
    ) -> torch.Tensor:
        """Can take any outputs,  ."""
        # Mask = 1 represents not masked points
        # Rescale output and target to [0, 1] for each image in batch
        output_min = torch.amin(model_output, dim=(1, 2, 3, 4), keepdim=True)
        output_max = torch.amax(model_output, dim=(1, 2, 3, 4), keepdim=True)
        target_min = torch.amin(target, dim=(1, 2, 3, 4), keepdim=True)
        target_max = torch.amax(target, dim=(1, 2, 3, 4), keepdim=True)

        rescaled_out = (model_output - output_min) / (output_max - output_min)
        rescaled_target = (target - target_min) / (target_max - target_min)

        # rescaled_out = rearrange(rescaled_out, "b x y z c -> b c x y z")
        # rescaled_target = rearrange(rescaled_target, "b x y z c -> b c x y z")
        rescaled_out = rescaled_out * (
            1 - mask
        )  # Make originally visible stuff 0, so that SSIM focuses on masked areas
        rescaled_target = rescaled_target * (1 - mask)

        ssim_loss = 1 - ssim(rescaled_out, rescaled_target, data_range=1, size_average=False, nonnegative_ssim=True)

        return torch.mean(ssim_loss)


class MAE_MS_SSIMLoss(AbstractLoss):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction="none")

    def forward(
        self, model_output: torch.Tensor, target: dict[str, torch.Tensor], mask: torch.Tensor
    ) -> torch.Tensor:
        """Can take any outputs,  ."""
        # Mask = 1 represents not masked points
        # Rescale output and target to [0, 1] for each image in batch
        output_min = torch.amin(model_output, dim=(1, 2, 3, 4), keepdim=True)
        output_max = torch.amax(model_output, dim=(1, 2, 3, 4), keepdim=True)
        target_min = torch.amin(target, dim=(1, 2, 3, 4), keepdim=True)
        target_max = torch.amax(target, dim=(1, 2, 3, 4), keepdim=True)

        rescaled_out = (model_output - output_min) / (output_max - output_min)
        rescaled_target = (target - target_min) / (target_max - target_min)

        # rescaled_out = rearrange(rescaled_out, "b x y z c -> b c x y z")
        # rescaled_target = rearrange(rescaled_target, "b x y z c -> b c x y z")

        ssim_loss = 1 - ms_ssim(rescaled_out, rescaled_target, data_range=1, size_average=False, win_size=7)

        return torch.mean(ssim_loss)


class MAE_MS_SSIMLoss_WithMask(AbstractLoss):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction="none")

    def forward(
        self, model_output: torch.Tensor, target: dict[str, torch.Tensor], mask: torch.Tensor
    ) -> torch.Tensor:
        """Can take any outputs,  ."""
        # Mask = 1 represents not masked points
        # Rescale output and target to [0, 1] for each image in batch
        output_min = torch.amin(model_output, dim=(1, 2, 3, 4), keepdim=True)
        output_max = torch.amax(model_output, dim=(1, 2, 3, 4), keepdim=True)
        target_min = torch.amin(target, dim=(1, 2, 3, 4), keepdim=True)
        target_max = torch.amax(target, dim=(1, 2, 3, 4), keepdim=True)

        rescaled_out = (model_output - output_min) / (output_max - output_min)
        rescaled_target = (target - target_min) / (target_max - target_min)

        # rescaled_out = rearrange(rescaled_out, "b x y z c -> b c x y z")
        # rescaled_target = rearrange(rescaled_target, "b x y z c -> b c x y z")

        rescaled_out = rescaled_out * (1 - mask)  # Set unmasked stuff 0, so MS SSIM focuses on masked areas
        rescaled_target = rescaled_target * (1 - mask)
        ssim_loss = 1 - ms_ssim(rescaled_out, rescaled_target, data_range=1, size_average=False, win_size=7)

        return torch.mean(ssim_loss)


class MSELoss_NoMask(AbstractLoss):
    def forward(
        self, model_output: torch.Tensor, target: dict[str, torch.Tensor], mask: torch.Tensor
    ) -> torch.Tensor:
        return self.loss(model_output, target)

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction="mean")


class L1Loss_NoMask(AbstractLoss):
    def forward(
        self, model_output: torch.Tensor, target: dict[str, torch.Tensor], mask: torch.Tensor
    ) -> torch.Tensor:
        return self.loss(model_output, target)

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss(reduction="mean")
