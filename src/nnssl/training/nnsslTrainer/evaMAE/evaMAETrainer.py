import torch
import os
import sys
from torch import nn
from typing import Tuple
from nnssl.training.nnsslTrainer.evaMAE.evaMAE_module import EvaMAE
from torch import autocast
from nnssl.utilities.helpers import dummy_context
from tqdm import tqdm
from valohai.config import is_running_in_valohai
from nnssl.experiment_planning.experiment_planners.plan import Plan
from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer
from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import BaseMAETrainer
import numpy as np
from nnssl.paths import nnssl_results
from torch import distributed as dist
from nnssl.training.logging.nnssl_logger_wandb import nnSSLLogger_wandb
from batchgenerators.utilities.file_and_folder_operations import join, isfile, save_json, maybe_mkdir_p, load_json
import wandb
from nnssl.utilities.helpers import empty_cache
class EvaMAETrainer(BaseMAETrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device,
    ):
        super(EvaMAETrainer, self).__init__(plan,
                         configuration_name,
                         fold,
                         pretrain_json,
                         device,
                         )

        self.output_folder_base = (
            join(
                nnssl_results,
                self.plan.dataset_name,
                self.__class__.__name__ + "__" + self.plan.plans_name + "__" + configuration_name,
                )
            if nnssl_results is not None
            else None
        )
        self.output_folder = join(self.output_folder_base, f"fold_{fold}")
        maybe_mkdir_p(self.output_folder)



        # use wandb nnssl logger
        self.use_wandb = True if self.local_rank == 0 else False
        group_name = (
                self.plan.dataset_name
                + "_"
                + self.__class__.__name__
                + "_"
                + self.plan.plans_name
                + "_"
                + self.configuration_name
        )
        if len(group_name) > 128:
            group_name = group_name[:128]
        wandb_init_args = {
            "dir": self.output_folder,
            "name": self.plan.dataset_name + "_fold" + str(fold),
            "group": group_name,
        }

        self.logger = nnSSLLogger_wandb(
            use_wandb=self.use_wandb,
            dataset_name=self.plan.dataset_name,
            wandb_init_args=wandb_init_args,
        )


        self.mask_ratio = self.config_plan['mask_ratio']
        self.vit_patch_size = self.config_plan['vit_patch_size']
        self.embed_dim = self.config_plan['embed_dim']
        self.encoder_eva_depth = self.config_plan['encoder_eva_depth']
        self.encoder_eva_numheads = self.config_plan['encoder_eva_numheads']
        self.decoder_eva_depth = self.config_plan['decoder_eva_depth']
        self.decoder_eva_numheads = self.config_plan['decoder_eva_numheads']
        self.batch_size_from_args = self.config_plan['bs']
        if self.config_plan['initial_lr'] is not None:
            self.initial_lr = self.config_plan['initial_lr']
        self._overwrite_batch_size()



    def _overwrite_batch_size(self):
        if not self.is_ddp:
            if self.batch_size_from_args is not None:
                # set the batch size from the arguments
                self.batch_size = self.batch_size_from_args
            else:
                # set batch size to what the plan says, leave oversample untouched
                self.batch_size = self.config_plan.batch_size

        else:
            # batch size is distributed over DDP workers and we need to change oversample_percent for each worker
            batch_sizes = []

            world_size = dist.get_world_size()
            my_rank = dist.get_rank()

            if self.batch_size_from_args is not None:
                # set the batch size from the arguments
                global_batch_size = self.batch_size_from_args
            else:
                global_batch_size = self.config_plan.batch_size
            assert global_batch_size >= world_size, (
                "Cannot run DDP if the batch size is smaller than the number of " "GPUs... Duh."
            )

            batch_size_per_GPU = np.ceil(global_batch_size / world_size).astype(int)

            for rank in range(world_size):
                if (rank + 1) * batch_size_per_GPU > global_batch_size:
                    batch_size = batch_size_per_GPU - ((rank + 1) * batch_size_per_GPU - global_batch_size)
                else:
                    batch_size = batch_size_per_GPU

                batch_sizes.append(batch_size)

            print("worker", my_rank, "batch_size", batch_sizes[my_rank])
            # self.print_to_log_file("worker", my_rank, "oversample", oversample_percents[my_rank])
            # self.print_to_log_file("worker", my_rank, "batch_size", batch_sizes[my_rank])

            self.batch_size = batch_sizes[my_rank]
    def _save_debug_information(self):
        # saving some debug information
        if self.local_rank == 0:
            dct = {}
            for k in self.__dir__():
                if not k.startswith("__"):
                    if not callable(getattr(self, k)) or k in [
                        "loss",
                    ]:
                        dct[k] = str(getattr(self, k))
                    elif k in [
                        "network",
                    ]:
                        dct[k] = str(getattr(self, k).__class__.__name__)
                    else:
                        # print(k)
                        pass
                if k in ["dataloader_train", "dataloader_val"]:
                    if hasattr(getattr(self, k), "generator"):
                        dct[k + ".generator"] = str(getattr(self, k).generator)
                    if hasattr(getattr(self, k), "num_processes"):
                        dct[k + ".num_processes"] = str(getattr(self, k).num_processes)
                    if hasattr(getattr(self, k), "transform"):
                        dct[k + ".transform"] = str(getattr(self, k).transform)
            import subprocess

            hostname = subprocess.getoutput(["hostname"])
            dct["hostname"] = hostname
            torch_version = torch.__version__
            if self.device.type == "cuda":
                gpu_name = torch.cuda.get_device_name()
                dct["gpu_name"] = gpu_name
                cudnn_version = torch.backends.cudnn.version()
            else:
                cudnn_version = "None"
            dct["device"] = str(self.device)
            dct["torch_version"] = torch_version
            dct["cudnn_version"] = cudnn_version
            save_json(dct, join(self.output_folder, "debug.json"))

            if self.use_wandb and self.local_rank == 0:
                self.logger.log_hypparams_to_wandb(self, dct)
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

    def build_architecture(self, config_plan, num_input_channels, num_output_channels) -> nn.Module:
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

    def on_train_start(self):
        if not self.was_initialized:
            self.initialize()

        maybe_mkdir_p(self.output_folder)

        self.print_plans()
        empty_cache(self.device)

        if self.is_ddp:
            dist.barrier()

        # dataloaders must be instantiated here because they need access to the training data which may not be present
        # when doing inference
        self.dataloader_train, self.dataloader_val = self.get_dataloaders()
        # Guarantee to only use data that is readable and not inf or nan

        # copy plans and dataset.json so that they can be used for restoring everything we need for inference
        save_json(asdict(self.plan), join(self.output_folder_base, "plans.json"), sort_keys=False)

        self._save_debug_information()
