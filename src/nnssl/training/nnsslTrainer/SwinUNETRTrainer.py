import torch
from torch import nn
from torch.optim import AdamW
from typing_extensions import override

from nnssl.architectures.build_architecture import build_network_architecture
from nnssl.architectures.swinunetr_architecture import SwinUNETRArchitecture
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.ssl_data.dataloading.swin_unetr_transform import SwinUNETRTransform
from nnssl.training.loss.swinunetr_loss import SwinUNETRLoss

from nnssl.training.lr_scheduler.polylr import PolyLRScheduler
from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer
from nnssl.utilities.collate_outputs import collate_outputs
from nnssl.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnssl.ssl_data.limited_len_wrapper import LimitedLenWrapper
from torch import autocast
from nnssl.utilities.helpers import dummy_context

from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor


import time
import numpy as np

# class Timer:
#     def __init__(self, name):
#         self.name = name
#         self.start_time = None
#         self.durations = []
#
#     def __enter__(self):
#         self.start_time = time.time()
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         elapsed_time = time.time() - self.start_time
#         self.durations.append(elapsed_time)
#
#     def print_avg_duration_in_ms(self):
#         x = 1e3*sum(self.durations)/len(self.durations)
#         print(f"{self.name}: {x}ms")
#         self.durations = []


class SwinUNETRTrainer(AbstractBaseTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)

        self.initial_lr = 4e-4
        self.weight_decay = 1e-5

        self.rec_loss_weight = 1
        self.contrast_loss_weight = 1
        self.rot_loss_weight = 1

        # self.num_iterations_per_epoch = 50

        # self.forward_t = Timer("forward")
        # self.loss_t = Timer("loss")
        # self.backward_t = Timer("backward")

    @override
    def build_loss(self):
        return SwinUNETRLoss(self.batch_size,
                             self.device,
                             self.rec_loss_weight,
                             self.contrast_loss_weight,
                             self.rot_loss_weight)

    @override
    def build_architecture(
        self, config_plan: ConfigurationPlan, num_input_channels: int, num_output_channels: int
    ) -> nn.Module:
        encoder = build_network_architecture(
            config_plan,
            num_input_channels,
            num_output_channels,
            encoder_only=True
        )
        architecture = SwinUNETRArchitecture(encoder, num_input_channels)

        # summary(architecture, input_size=(96,)*3, batch_size=4)

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
    def configure_optimizers(self):
        optimizer = AdamW(
            params=self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay
        )
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

        return optimizer, lr_scheduler

    @override
    def train_step(self, batch: dict) -> dict:

        imgs1_rotated, imgs2_rotated = batch["imgs_rotated"]
        rotations1, rotations2 = batch["rotations"]
        imgs1_rotated_cutout, imgs2_rotated_cutout = batch["imgs_rotated_cutout"]

        imgs_rotated = torch.cat([imgs1_rotated, imgs2_rotated], dim=0)
        rotations = torch.cat([rotations1, rotations2], dim=0)
        imgs_rotated_cutout = torch.cat([imgs1_rotated_cutout, imgs2_rotated_cutout], dim=0)

        imgs_rotated = imgs_rotated.to(self.device, non_blocking=True)
        rotations = rotations.to(self.device, non_blocking=True)
        imgs_rotated_cutout = imgs_rotated_cutout.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            # with self.forward_t:
            rotations_pred, contrast_pred, reconstructions = self.network(imgs_rotated_cutout)
            # contrast1_pred, contrast2_pred = contrast_pred[:self.batch_size], contrast_pred[self.batch_size:]
            # with self.loss_t:
            l = self.loss(rotations_pred, rotations, contrast_pred, reconstructions, imgs_rotated)

        if self.grad_scaler is not None:
            # with self.backward_t:
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

    # def on_train_epoch_end(self, train_outputs: list[dict]):
    #     self.interrupt_at_nans(train_outputs)
    #     outputs = collate_outputs(train_outputs)
    #
    #     self.forward_t.print_avg_duration_in_ms()
    #     self.backward_t.print_avg_duration_in_ms()
    #     self.loss_t.print_avg_duration_in_ms()
    #
    #     loss_here = np.mean(outputs["loss"])
    #     self.logger.log("train_losses", loss_here, self.current_epoch)


    @override
    def validation_step(self, batch: dict) -> dict:
        imgs1_rotated, imgs2_rotated = batch["imgs_rotated"]
        rotations1, rotations2 = batch["rotations"]
        imgs1_rotated_cutout, imgs2_rotated_cutout = batch["imgs_rotated_cutout"]

        imgs_rotated = torch.cat([imgs1_rotated, imgs2_rotated], dim=0)
        rotations = torch.cat([rotations1, rotations2], dim=0)
        imgs_rotated_cutout = torch.cat([imgs1_rotated_cutout, imgs2_rotated_cutout], dim=0)

        imgs_rotated = imgs_rotated.to(self.device, non_blocking=True)
        rotations = rotations.to(self.device, non_blocking=True)
        imgs_rotated_cutout = imgs_rotated_cutout.to(self.device, non_blocking=True)

        with torch.no_grad():
            with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                rotations_pred, contrast_pred, reconstructions = self.network(imgs_rotated_cutout)
                contrast1_pred, contrast2_pred = contrast_pred[:self.batch_size], contrast_pred[self.batch_size:]
                l = self.loss(rotations_pred, rotations, contrast1_pred, contrast2_pred, reconstructions, imgs_rotated)

        return {"loss": l.detach().cpu().numpy()}

    @staticmethod
    def get_training_transforms() -> AbstractTransform:
        tr_transforms = []

        tr_transforms.append(SwinUNETRTransform())
        tr_transforms.append(NumpyToTensor(cast_to="float", keys=["imgs_rotated", "imgs_rotated_cutout"]))
        tr_transforms.append(NumpyToTensor(cast_to="long", keys="rotations"))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms

    @staticmethod
    def get_validation_transforms() -> AbstractTransform:
        return SwinUNETRTrainer.get_training_transforms()


class SwinUNETRTrainer_orig(SwinUNETRTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 2
        plan.configurations[configuration_name].patch_size = (128, 128, 128)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)


class SwinUNETRTrainer_BS6(SwinUNETRTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 6
        plan.configurations[configuration_name].patch_size = (160, 160, 160)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)


class SwinUNETRTrainer_BS2(SwinUNETRTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 2
        plan.configurations[configuration_name].patch_size = (160, 160, 160)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)

class SwinUNETRTrainer_BS3(SwinUNETRTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 3
        plan.configurations[configuration_name].patch_size = (160, 160, 160)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)

class SwinUNETRTrainer_BS4(SwinUNETRTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 4
        plan.configurations[configuration_name].patch_size = (160, 160, 160)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)


class SwinUNETRTrainer_two_forward_passes(SwinUNETRTrainer):
    @override
    def train_step(self, batch: dict) -> dict:
        imgs1_rotated, imgs2_rotated = batch["imgs_rotated"]
        rotations1, rotations2 = batch["rotations"]
        imgs1_rotated_cutout, imgs2_rotated_cutout = batch["imgs_rotated_cutout"]

        imgs1_rotated = imgs1_rotated.to(self.device, non_blocking=True)
        imgs2_rotated = imgs2_rotated.to(self.device, non_blocking=True)
        rotations1 = rotations1.to(self.device, non_blocking=True)
        rotations2 = rotations2.to(self.device, non_blocking=True)
        imgs1_rotated_cutout = imgs1_rotated_cutout.to(self.device, non_blocking=True)
        imgs2_rotated_cutout = imgs2_rotated_cutout.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            rotations1_pred, contrast1_pred, reconstructions1 = self.network(imgs1_rotated_cutout)
            rotations2_pred, contrast2_pred, reconstructions2 = self.network(imgs2_rotated_cutout)

            rotations_pred = torch.cat([rotations1_pred, rotations2_pred], dim=0)
            rotations = torch.cat([rotations1, rotations2], dim=0)
            reconstructions = torch.cat([reconstructions1, reconstructions2], dim=0)
            imgs_rotated = torch.cat([imgs1_rotated, imgs2_rotated], dim=0)

            l = self.loss(rotations_pred, rotations, contrast1_pred, contrast2_pred, reconstructions, imgs_rotated)

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

    @override
    def validation_step(self, batch: dict) -> dict:
        imgs1_rotated, imgs2_rotated = batch["imgs_rotated"]
        rotations1, rotations2 = batch["rotations"]
        imgs1_rotated_cutout, imgs2_rotated_cutout = batch["imgs_rotated_cutout"]

        imgs1_rotated = imgs1_rotated.to(self.device, non_blocking=True)
        imgs2_rotated = imgs2_rotated.to(self.device, non_blocking=True)
        rotations1 = rotations1.to(self.device, non_blocking=True)
        rotations2 = rotations2.to(self.device, non_blocking=True)
        imgs1_rotated_cutout = imgs1_rotated_cutout.to(self.device, non_blocking=True)
        imgs2_rotated_cutout = imgs2_rotated_cutout.to(self.device, non_blocking=True)

        with torch.no_grad():
            with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                rotations1_pred, contrast1_pred, reconstructions1 = self.network(imgs1_rotated_cutout)
                rotations2_pred, contrast2_pred, reconstructions2 = self.network(imgs2_rotated_cutout)

                rotations_pred = torch.cat([rotations1_pred, rotations2_pred], dim=0)
                rotations = torch.cat([rotations1, rotations2], dim=0)
                reconstructions = torch.cat([reconstructions1, reconstructions2], dim=0)
                imgs_rotated = torch.cat([imgs1_rotated, imgs2_rotated], dim=0)

                l = self.loss(rotations_pred, rotations, contrast1_pred, contrast2_pred, reconstructions, imgs_rotated)

        return {"loss": l.detach().cpu().numpy()}


