import torch
import os

from torch import nn
from typing import Tuple
from nnssl.training.nnsslTrainer.evaMAE.evaMAE_module import EvaMAE
from torch import autocast
from nnssl.utilities.helpers import dummy_context
from batchgenerators.utilities.file_and_folder_operations import join
from tqdm import tqdm
from valohai.config import is_running_in_valohai
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
        device: torch.device, 
        mask_ratio,
        vit_patch_size,
        embed_dim,
        encoder_eva_depth,
        encoder_eva_numheads,
        decoder_eva_depth,
        decoder_eva_numheads,
        bs,
    ):
        #import IPython
        #IPython.embed()
        super(EvaMAETrainer, self).__init__(plan,
                         configuration_name,
                         fold,
                         pretrain_json,
                         device,
                         mask_ratio,
                         vit_patch_size,
                         embed_dim,
                         encoder_eva_depth,
                         encoder_eva_numheads,
                         decoder_eva_depth,
                         decoder_eva_numheads,
                         bs,
                         )
        
        self.mask_ratio = mask_ratio
        self.vit_patch_size = vit_patch_size
        self.embed_dim = embed_dim
        self.encoder_eva_depth = encoder_eva_depth
        self.encoder_eva_numheads = encoder_eva_numheads
        self.decoder_eva_depth = decoder_eva_depth
        self.decoder_eva_numheads = decoder_eva_numheads
    
    @staticmethod
    def create_mask(keep_indices: torch.Tensor, image_size: Tuple[int, int, int], patch_size: Tuple[int, int, int]) -> torch.Tensor:
        """
        Create a mask tensor (1 for unmasked, 0 for masked) based on keep_indices.

        Args:
            keep_indices (torch.Tensor): Tensor of shape (B, num_kept_patches) indicating retained patches.
            image_size (Tuple[int, int, int]): Size of the full image (D, H, W).
            patch_size (Tuple[int, int, int]): Size of each patch (D_patch, H_patch, W_patch).

        Returns:
            torch.Tensor: Mask tensor of shape (B, 1, D, H, W) with 1 for unmasked and 0 for masked.
        """
        B, num_kept_patches = keep_indices.shape
        D, H, W = image_size
        D_patch, H_patch, W_patch = patch_size

        # Calculate the number of patches along each dimension
        num_patches_d = D // D_patch
        num_patches_h = H // H_patch
        num_patches_w = W // W_patch
        num_patches = num_patches_d * num_patches_h * num_patches_w

        # Create a flat mask of 0s with shape (B, num_patches)
        flat_mask = torch.zeros(B, num_patches, device=keep_indices.device)

        # Set retained patches to 1
        flat_mask.scatter_(1, keep_indices, 1)

        # Reshape to patch grid and expand to full image size
        mask = flat_mask.view(B, num_patches_d, num_patches_h, num_patches_w)
        mask = mask.repeat_interleave(D_patch, dim=1).repeat_interleave(H_patch, dim=2).repeat_interleave(W_patch, dim=3)
        mask = mask.unsqueeze(1)  # Add channel dimension (B, 1, D, H, W)
        return mask

    def build_architecture(self, *args, **kwargs) -> nn.Module:
        network = EvaMAE(
            input_channels=1,
            embed_dim=self.embed_dim,
            patch_embed_size=self.vit_patch_size,
            output_channels=1,
            input_shape=self.config_plan.patch_size,
            encoder_eva_depth=self.encoder_eva_depth,
            encoder_eva_numheads=self.encoder_eva_numheads,
            decoder_eva_depth=self.decoder_eva_depth,
            decoder_eva_numheads=self.decoder_eva_numheads,
            patch_drop_rate=self.mask_ratio
        )
        return network
    
    def on_validation_epoch_start(self):
        #self.network.eval()
        pass
    
    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        data = data.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        # Autocast for CUDA device
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            # Forward pass with PatchDropout
            output, keep_indices = self.network(data)
            mask = self.create_mask(keep_indices, self.config_plan.patch_size, self.vit_patch_size)
            # Calculate loss considering kept patches
            l = self.loss(output, data, mask)

        # Backward pass and optimization
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {"loss": l.detach().cpu().numpy()}


    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        data = data.to(self.device, non_blocking=True)

        # Autocast for CUDA device
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            # Forward pass with PatchDropout
            output, keep_indices = self.network(data)
            mask = self.create_mask(keep_indices, self.config_plan.patch_size, self.vit_patch_size)
            # Calculate loss considering kept patches
            l = self.loss(output, data, mask)

        return {"loss": l.detach().cpu().numpy()}
    
    def run_training(self):
        try:
            self.on_train_start()
            if self.local_rank == 0:
                self.log_qualitative_reconstruction_step()  # Do a quick test everything works.
            for epoch in range(self.current_epoch, self.num_epochs):
                self.on_epoch_start()

                self.on_train_epoch_start()
                train_outputs = []
                for batch_id in tqdm(
                    range(self.num_iterations_per_epoch),
                    desc=f"Epoch {epoch}",
                    disable=True if (("LSF_JOBID" in os.environ) or is_running_in_valohai()) else False,
                ):
                    train_outputs.append(self.train_step(next(self.dataloader_train)))
                self.on_train_epoch_end(train_outputs)

                with torch.no_grad():
                    self.on_validation_epoch_start()
                    val_outputs = []
                    for batch_id in tqdm(range(self.num_val_iterations_per_epoch)):
                        val_outputs.append(self.validation_step(next(self.dataloader_val)))
                    self.on_validation_epoch_end(val_outputs)

                    # ------------------------ Maybe Log qualitative recon ----------------------- #
                    if (self.current_epoch + 1) % self.save_imgs_every_n_epochs == 0:
                        if self.local_rank == 0:
                            self.log_qualitative_reconstruction_step()
                            # self.save_checkpoint(
                            #     join(self.output_folder, f"checkpoint_epoch_{self.current_epoch}.pth"), live_upload=True
                            # )

                self.on_epoch_end()

            self.on_train_end()
        except KeyboardInterrupt:
            self.print_to_log_file("Keyboard interrupt. Exiting gracefully.")
            self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))
            raise KeyboardInterrupt
        
    def log_qualitative_reconstruction_step(self):
        """Log qualitative reconstructions for each sample in the validation dataloader."""
        with torch.no_grad():
            for batch_id in range(10):
                data = self.recon_dataloader[batch_id]["data"]
                data = data.to(self.device, non_blocking=True)

                # Forward pass with PatchDropout to get reconstruction and keep_indices
                with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                    reconstruction, keep_indices = self.network(data)

                mask = self.create_mask(keep_indices, self.config_plan.patch_size, self.vit_patch_size)

                # Compute loss for each sample in the batch
                losses = [
                    self.loss(reconstruction[i : i + 1], data[i : i + 1], mask[i : i + 1])
                    for i in range(reconstruction.shape[0])
                ]

                # Log the images with slices, reconstruction, and losses
                self.log_img_slices(data, reconstruction, mask, losses, batch_id)

        return
