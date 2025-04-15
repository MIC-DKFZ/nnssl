import os
import random
from itertools import combinations

import numpy as np
import torch
from math import prod

from batchgenerators.utilities.file_and_folder_operations import join
from torch import nn
from tqdm import tqdm
from typing_extensions import override

from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan
from nnssl.architectures.get_network_by_name import get_network_by_name
from nnssl.architectures.pclrv2_architecture import PCRLv2Architecture
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.ssl_data.dataloading.pcrlv2_transform import PCLRv2Transform, Shape3D
from nnssl.ssl_data.dataloading.swin_unetr_transform import SwinUNETRTransform
from nnssl.training.loss.pcrlv2_loss import PCRLv2Loss

from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer
from nnssl.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnssl.ssl_data.limited_len_wrapper import LimitedLenWrapper
from torch import autocast
from nnssl.utilities.helpers import dummy_context

from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor


class PCRLv2Trainer(AbstractBaseTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
        # Our network has 5 downsampling stages, we need a minimum size of 2⁵=32 per axis. Although we could set
        # the global_input_size to our standard (160, 160, 160), the orig config *never* upsamples their
        # global crops. We try to scale the config while keeping it reasonable and close to the original.
        global_patch_sizes: tuple[Shape3D] = (
            (96, 96, 96),
            (128, 128, 96),
            (128, 128, 128),
            (160, 160, 128),
            (160, 160, 160),
        ),
        global_input_size: Shape3D = (128, 128, 128),
        local_patch_sizes: tuple[Shape3D] = ((32, 32, 32), (64, 64, 32), (64, 64, 64)),
        local_input_size: Shape3D = (64, 64, 64),
        # orig config
        # global_patch_sizes: tuple[Shape3D] = ((64, 64, 32), (96, 96, 64), (96, 96, 96), (112, 112, 64)),
        # global_input_size: Shape3D = (64, 64, 32),
        # local_patch_sizes: tuple[Shape3D] = ((8, 8, 8), (16, 16, 16), (32, 32, 16), (32, 32, 32)),
        # local_input_size: Shape3D = (16, 16, 16),
        num_locals: int = 6,
        min_IoU: float = 0.3,
    ):
        # We want the dataloader to give us a patch_size big enough, to accommodate for the largest patch size
        # for each axis, while making it possible to have different overlapping volumes and still allow the anatomy
        # to be a large part of the patch
        plan.configurations[configuration_name].patch_size = (180, 180, 180)

        super().__init__(plan, configuration_name, fold, pretrain_json, device)

        self.global_patch_sizes = global_patch_sizes
        self.global_input_size = global_input_size
        self.local_patch_sizes = local_patch_sizes
        self.local_input_size = local_input_size
        self.num_locals = num_locals
        self.min_IoU = min_IoU

    @override
    def build_loss(self):
        return PCRLv2Loss(self.num_mid_stages, self.num_locals)

    @override
    def build_architecture_and_adaptation_plan(
        self, config_plan: ConfigurationPlan, num_input_channels: int, num_output_channels: int
    ) -> nn.Module:
        network = get_network_by_name(config_plan, "NoSkipResEncL", num_input_channels, num_output_channels)
        architecture = PCRLv2Architecture(network)
        self.num_mid_stages = len(architecture.features_per_mid_stage)
        raise NotImplementedError("Missing adaptation plan")
        return architecture

    @override
    def get_dataloaders(self):

        tr_transforms = self.get_training_transforms()
        val_transforms = self.get_validation_transforms()

        dl_tr, dl_val = self.get_plain_dataloaders(initial_patch_size=self.config_plan.patch_size)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, tr_transforms)
            mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
        else:
            mt_gen_train = LimitedLenWrapper(
                self.num_iterations_per_epoch,
                data_loader=dl_tr,
                transform=tr_transforms,
                num_processes=allowed_num_processes,
                num_cached=6,
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.02,
            )
            mt_gen_val = LimitedLenWrapper(
                self.num_val_iterations_per_epoch,
                data_loader=dl_val,
                transform=val_transforms,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=3,
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.02,
            )
        return mt_gen_train, mt_gen_val

    @override
    def train_step(self, batch: dict) -> dict:

        aug_global_crops_A = batch[
            "aug_global_crops_A"
        ]  # [B,              C, X_global_input_size, Y_global_input_size, Z_global_input_size]
        global_crops_A = batch[
            "global_crops_A"
        ]  # [B,              C, X_global_input_size, Y_global_input_size, Z_global_input_size]
        aug_global_crops_B = batch[
            "aug_global_crops_B"
        ]  # [B,              C, X_global_input_size, Y_global_input_size, Z_global_input_size]
        aug_local_crops = batch[
            "aug_local_crops"
        ]  # [(B*num_locals), C, X_local_input_size,  Y_local_input_size,  Z_local_input_size ]

        aug_global_crops_A = aug_global_crops_A.to(self.device, non_blocking=True)
        global_crops_A = global_crops_A.to(self.device, non_blocking=True)
        aug_global_crops_B = aug_global_crops_B.to(self.device, non_blocking=True)
        aug_local_crops = aug_local_crops.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            reconstructions_A, embeddings_A, mid_reconstructions_A = self.network(aug_global_crops_A)
            embeddings_B = self.network(aug_global_crops_B, embeddings_only=True)
            local_embeddings = self.network(aug_local_crops, embeddings_only=True)

            rec_l, mid_rec_l, g_sim_l, l_sim_l = self.loss(
                reconstructions_A, mid_reconstructions_A, global_crops_A, embeddings_A, embeddings_B, local_embeddings
            )
            l = rec_l + mid_rec_l + g_sim_l + l_sim_l

            # l = self.loss(reconstructions_A, mid_reconstructions_A, global_crops_A, embeddings_A, embeddings_B,
            #               local_embeddings)

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

        return (
            {"loss": l.detach().cpu().numpy()},
            rec_l.detach().cpu().numpy(),
            mid_rec_l.detach().cpu().numpy(),
            g_sim_l.detach().cpu().numpy(),
            l_sim_l.detach().cpu().numpy(),
        )
        # return {"loss": np.array(0)}

    def run_training(self):
        try:
            self.on_train_start()

            for epoch in range(self.current_epoch, self.num_epochs):
                self.on_epoch_start()

                self.on_train_epoch_start()
                train_outputs = []

                rec_ls, mid_rec_ls, g_sim_ls, l_sim_ls = [], [], [], []

                for batch_id in tqdm(
                    range(self.num_iterations_per_epoch),
                    desc=f"Epoch {epoch}",
                    disable=True if (("LSF_JOBID" in os.environ) or ("SLURM_JOB_ID" in os.environ)) else False,
                ):
                    l, rec_l, mid_rec_l, g_sim_l, l_sim_l = self.train_step(next(self.dataloader_train))
                    rec_ls.append(rec_l)
                    mid_rec_ls.append(mid_rec_l)
                    g_sim_ls.append(g_sim_l)
                    l_sim_ls.append(l_sim_l)
                    train_outputs.append(l)

                    # train_outputs.append(self.train_step(next(self.dataloader_train)))

                self.print_to_log_file(
                    f"Recon Loss: {np.mean(rec_ls).item()}",
                    f" | Mid Recon Loss: {np.mean(mid_rec_ls).item()}"
                    f" | Global Sim Loss: {np.mean(g_sim_ls).item()}"
                    f" | Local Sim Loss: {np.mean(l_sim_ls).item()}",
                )

                self.on_train_epoch_end(train_outputs)

                with torch.no_grad():
                    self.on_validation_epoch_start()
                    val_outputs = []
                    for batch_id in range(self.num_val_iterations_per_epoch):
                        val_outputs.append(self.validation_step(next(self.dataloader_val)))
                        # val_outputs.append(self.validation_step(next(self.dataloader_val)))
                    self.on_validation_epoch_end(val_outputs)

                if self.exit_training_flag:
                    # This is a signal that we need to resubmit, so we break the loop and exit gracefully
                    print("Finished last epoch before restart.")
                    self.print_to_log_file("Finished last epoch before restart.")
                    raise KeyboardInterrupt

                self.on_epoch_end()

            self.on_train_end()
        except KeyboardInterrupt:
            print("Keyboard interrupt.")
            self.print_to_log_file("Keyboard interrupt. Exiting gracefully.")
            self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))
            raise KeyboardInterrupt

    @override
    def validation_step(self, batch: dict) -> dict:
        aug_global_crops_A = batch[
            "aug_global_crops_A"
        ]  # [B,              C, X_global_input_size, Y_global_input_size, Z_global_input_size]
        global_crops_A = batch[
            "global_crops_A"
        ]  # [B,              C, X_global_input_size, Y_global_input_size, Z_global_input_size]
        aug_global_crops_B = batch[
            "aug_global_crops_B"
        ]  # [B,              C, X_global_input_size, Y_global_input_size, Z_global_input_size]
        aug_local_crops = batch[
            "aug_local_crops"
        ]  # [(B*num_locals), C, X_local_input_size,  Y_local_input_size,  Z_local_input_size ]

        aug_global_crops_A = aug_global_crops_A.to(self.device, non_blocking=True)
        global_crops_A = global_crops_A.to(self.device, non_blocking=True)
        aug_global_crops_B = aug_global_crops_B.to(self.device, non_blocking=True)
        aug_local_crops = aug_local_crops.to(self.device, non_blocking=True)

        with torch.no_grad():
            with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                reconstructions_A, embeddings_A, mid_reconstructions_A = self.network(aug_global_crops_A)
                embeddings_B = self.network(aug_global_crops_B, embeddings_only=True)
                local_embeddings = self.network(aug_local_crops, embeddings_only=True)
                rec_l, mid_rec_l, g_sim_l, l_sim_l = self.loss(
                    reconstructions_A,
                    mid_reconstructions_A,
                    global_crops_A,
                    embeddings_A,
                    embeddings_B,
                    local_embeddings,
                )
                l = rec_l + mid_rec_l + g_sim_l + l_sim_l
        return {"loss": l.detach().cpu().numpy()}

    def get_training_transforms(self) -> AbstractTransform:
        tr_transforms = Compose(
            [
                PCLRv2Transform(
                    self.global_patch_sizes,
                    self.global_input_size,
                    self.local_patch_sizes,
                    self.local_input_size,
                    self.num_locals,
                    self.min_IoU,
                ),
                NumpyToTensor(
                    keys=["aug_global_crops_A", "global_crops_A", "aug_global_crops_B", "aug_local_crops"],
                    cast_to="float",
                ),
            ]
        )
        return tr_transforms

    def get_validation_transforms(self) -> AbstractTransform:
        return self.get_training_transforms()


####################################################################
############################# VARIANTS #############################
####################################################################


class PCRLv2Trainer_test(PCRLv2Trainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device, global_input_size=(96, 96, 96))
        self.total_batch_size = 2
        self.num_iterations_per_epoch = 20
        self.num_val_iterations_per_epoch = 5
        self.initial_lr = 1e-3


class PCRLv2Trainer_BS8(PCRLv2Trainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 8
