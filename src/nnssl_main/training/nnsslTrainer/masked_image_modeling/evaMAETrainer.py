import torch
import os
from torch import nn
from torch._dynamo import OptimizedModule
from typing import Tuple, Union
from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans
from nnssl.architectures.evaMAE_module import EvaMAE
from torch import autocast
from nnssl.utilities.helpers import dummy_context
from tqdm import tqdm
from nnssl.experiment_planning.experiment_planners.plan import Plan
from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import BaseMAETrainer
from nnssl.training.lr_scheduler.warmup import Lin_incr_LRScheduler, PolyLRScheduler_offset
import numpy as np
from nnssl.paths import nnssl_results
from torch import distributed as dist
from nnssl.training.logging.nnssl_logger_wandb import nnSSLLogger_wandb
from batchgenerators.utilities.file_and_folder_operations import join, save_json, maybe_mkdir_p
from torch.nn.parallel import DistributedDataParallel as DDP
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
        super(EvaMAETrainer, self).__init__(
            plan,
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
        ###settings taken from fabi
        self.drop_path_rate = 0.2
        self.attention_drop_rate = 0.2
        self.grad_clip = 1
        self.initial_lr = 3e-5
        self.weight_decay = 5e-3
        self.enable_deep_supervision = False
        self.warmup_duration_whole_net = 50  # lin increase whole network
        self.training_stage = None

        self.mask_ratio = self.config_plan["mask_ratio"]
        self.vit_patch_size = self.config_plan["vit_patch_size"]
        self.embed_dim = self.config_plan["embed_dim"]
        self.encoder_eva_depth = self.config_plan["encoder_eva_depth"]
        self.encoder_eva_numheads = self.config_plan["encoder_eva_numheads"]
        self.decoder_eva_depth = self.config_plan["decoder_eva_depth"]
        self.decoder_eva_numheads = self.config_plan["decoder_eva_numheads"]
        self.batch_size_from_args = self.config_plan["batch_size"]
        if self.config_plan["initial_lr"] is not None:
            self.initial_lr = self.config_plan["initial_lr"]
        if self.config_plan["attention_drop_rate"] is not None:
            self.attention_drop_rate = self.config_plan["attention_drop_rate"]
        self._overwrite_batch_size()

    def configure_optimizers(self, stage: str = "warmup_all"):
        assert stage in ["warmup_all", "train"]

        if self.training_stage == stage:
            return self.optimizer, self.lr_scheduler

        if isinstance(self.network, DDP):
            params = self.network.module.parameters()
        else:
            params = self.network.parameters()

        if stage == "warmup_all":
            self.print_to_log_file("train whole net, warmup")
            optimizer = torch.optim.AdamW(
                params, self.initial_lr, weight_decay=self.weight_decay, amsgrad=False, betas=(0.9, 0.98), fused=True
            )
            lr_scheduler = Lin_incr_LRScheduler(optimizer, self.initial_lr, self.warmup_duration_whole_net)
            self.print_to_log_file(f"Initialized warmup_all optimizer and lr_scheduler at epoch {self.current_epoch}")
        else:
            self.print_to_log_file("train whole net, default schedule")
            if self.training_stage == "warmup_all":
                # we can keep the existing optimizer and don't need to create a new one. This will allow us to keep
                # the accumulated momentum terms which already point in a useful driection
                optimizer = self.optimizer
            else:
                optimizer = torch.optim.AdamW(
                    params,
                    self.initial_lr,
                    weight_decay=self.weight_decay,
                    amsgrad=False,
                    betas=(0.9, 0.98),
                    fused=True,
                )
            lr_scheduler = PolyLRScheduler_offset(
                optimizer, self.initial_lr, self.num_epochs, self.warmup_duration_whole_net
            )
            self.print_to_log_file(f"Initialized train optimizer and lr_scheduler at epoch {self.current_epoch}")
        self.training_stage = stage
        empty_cache(self.device)
        return optimizer, lr_scheduler

    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("warmup_all")
        elif self.current_epoch == self.warmup_duration_whole_net:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("train")

        super().on_train_epoch_start()

    def _overwrite_batch_size(self):
        if not self.is_ddp:
            if self.batch_size_from_args is not None:
                # set the batch size from the arguments
                self.batch_size = self.batch_size_from_args
            else:
                # set batch size to what the plan says, leave oversample untouched
                self.batch_size = self.total_batch_size

        else:
            # batch size is distributed over DDP workers and we need to change oversample_percent for each worker
            batch_sizes = []

            world_size = dist.get_world_size()
            my_rank = dist.get_rank()

            if self.batch_size_from_args is not None:
                # set the batch size from the arguments
                global_batch_size = self.batch_size_from_args
            else:
                global_batch_size = self.total_batch_size
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
    def create_mask(
        keep_indices: torch.Tensor, image_size: Tuple[int, int, int], patch_size: Tuple[int, int, int]
    ) -> torch.Tensor:
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
        mask = (
            mask.repeat_interleave(D_patch, dim=1).repeat_interleave(H_patch, dim=2).repeat_interleave(W_patch, dim=3)
        )
        mask = mask.unsqueeze(1)  # Add channel dimension (B, 1, D, H, W)
        return mask

    def build_architecture_and_adaptation_plan(
        self, config_plan, num_input_channels, num_output_channels
    ) -> nn.Module:
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
            patch_drop_rate=self.mask_ratio,
            drop_path_rate=self.drop_path_rate,
            attn_drop_rate=self.attention_drop_rate,
        )
        adapt_plan = AdaptationPlan(
            architecture_plans=ArchitecturePlans("PrimusM"),
            pretrain_plan=self.plan,
            pretrain_num_input_channels=1,
            recommended_downstream_patchsize=self.recommended_downstream_patchsize,
            key_to_encoder="eva",
            key_to_stem="down_projection",
            key_to_in_proj=("down_projection.proj",),
            key_to_lpe="eva.pos_embed",
        )
        raise NotImplementedError("Current AdaptationPlan is not correct")
        return network, adapt_plan

    def on_validation_epoch_start(self):
        # self.network.eval()
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
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
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
            for epoch in range(self.current_epoch, self.num_epochs):
                self.on_epoch_start()

                self.on_train_epoch_start()
                train_outputs = []
                for batch_id in tqdm(
                    range(self.num_iterations_per_epoch),
                    desc=f"Epoch {epoch}",
                    disable=True if ("LSF_JOBID" in os.environ) else False,
                ):
                    train_outputs.append(self.train_step(next(self.dataloader_train)))
                self.on_train_epoch_end(train_outputs)

                with torch.no_grad():
                    self.on_validation_epoch_start()
                    val_outputs = []
                    for batch_id in tqdm(range(self.num_val_iterations_per_epoch)):
                        val_outputs.append(self.validation_step(next(self.dataloader_val)))
                    self.on_validation_epoch_end(val_outputs)

                if self.exit_training_flag:
                    # This is a signal that we need to resubmit, so we break the loop and exit gracefully
                    print("Finished last epoch before restart.")
                    self.print_to_log_file("Finished last epoch before restart.")
                    raise KeyboardInterrupt
                self.on_epoch_end()

            self.on_train_end()
        except KeyboardInterrupt:
            self.print_to_log_file("Keyboard interrupt. Exiting gracefully.")
            self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))
            raise KeyboardInterrupt

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint["network_weights"].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith("module."):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint["init_args"]

        self.current_epoch = checkpoint["current_epoch"]
        min_epoch = self.logger.load_checkpoint(checkpoint["logging"])
        # Apparently the val log is not written correctly when we currently save the checkpoint.
        self.current_epoch = min_epoch
        self._best_ema = checkpoint["_best_ema"]

        # messing with state dict naming schemes. Facepalm.
        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)

        # it's fine to do this every time we load because configure_optimizers will be a no-op if the correct optimizer
        # and lr scheduler are already set up
        if self.current_epoch < self.warmup_duration_whole_net:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("warmup_all")
        else:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("train")

        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.grad_scaler is not None:
            if checkpoint["grad_scaler_state"] is not None:
                self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state"])


class EvaMAETrainerDEBUG(EvaMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device,
    ):
        super(EvaMAETrainerDEBUG, self).__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
        )

        self.num_iterations_per_epoch = 1
        self.num_val_iterations_per_epoch = 1


class EvaMAETrainer2kEpochs(EvaMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device,
    ):
        super(EvaMAETrainerDEBUG, self).__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
        )

        self.num_epochs = 2000


class EvaMAETrainer4kEpochs(EvaMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device,
    ):
        super(EvaMAETrainerDEBUG, self).__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
        )

        self.num_epochs = 4000
