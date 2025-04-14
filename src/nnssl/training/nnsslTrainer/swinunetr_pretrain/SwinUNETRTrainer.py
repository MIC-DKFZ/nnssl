import torch
from torch import nn
from typing_extensions import override

from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans
from nnssl.architectures.get_network_by_name import get_network_by_name
from nnssl.architectures.swinunetr_architecture import SwinUNETRArchitecture
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.ssl_data.dataloading.swin_unetr_transform import SwinUNETRTransform
from nnssl.training.loss.swinunetr_loss import SwinUNETRLoss

from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer
from nnssl.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnssl.ssl_data.limited_len_wrapper import LimitedLenWrapper
from torch import autocast
from nnssl.utilities.helpers import dummy_context

from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import save_json


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
        self.config_plan.patch_size = (160, 160, 160)
        # from paper
        # self.initial_lr = 4e-4
        # self.weight_decay = 1e-5

        self.rec_loss_weight = 1
        self.contrast_loss_weight = 1
        self.rot_loss_weight = 1

    @override
    def build_loss(self):
        return SwinUNETRLoss(
            self.batch_size, self.device, self.rec_loss_weight, self.contrast_loss_weight, self.rot_loss_weight
        )

    @override
    def build_architecture_and_adaptation_plan(
        self, config_plan: ConfigurationPlan, num_input_channels: int, num_output_channels: int
    ) -> nn.Module:
        encoder = get_network_by_name(
            config_plan,
            "ResEncL",
            num_input_channels,
            num_output_channels,
            encoder_only=True,
        )
        architecture = SwinUNETRArchitecture(encoder, self.num_output_channels)
        # ------------------------------ Adaptation Plan ----------------------------- #
        adapt_plan = AdaptationPlan(
            architecture_plans=ArchitecturePlans("ResEncL"),
            pretrain_plan=self.plan,
            recommended_downstream_patchsize=self.recommended_downstream_patchsize,
            pretrain_num_input_channels=num_input_channels,
            key_to_encoder="encoder.stages",
            key_to_stem="encoder.stem",
            keys_to_in_proj=("encoder.stem.convs.0.conv", "encoder.stem.convs.0.all_modules.0"),
        )
        return architecture, adapt_plan

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

    # @override
    # def configure_optimizers(self):
    #     optimizer = AdamW(
    #         params=self.network.parameters(),
    #         lr=self.initial_lr,
    #         weight_decay=self.weight_decay
    #     )
    #     lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
    #
    #     return optimizer, lr_scheduler

    @override
    def train_step(self, batch: dict) -> dict:

        imgs1_rotated, imgs2_rotated = batch["imgs_rotated"]
        rotations1, rotations2 = batch["rotations"]
        imgs1_rotated_cutout, imgs2_rotated_cutout = batch["imgs_rotated_cutout"]
        # print(f"rank: {self.local_rank}", imgs1_rotated_cutout.shape)

        imgs_rotated = torch.cat([imgs1_rotated, imgs2_rotated], dim=0)
        rotations = torch.cat([rotations1, rotations2], dim=0)
        imgs_rotated_cutout = torch.cat([imgs1_rotated_cutout, imgs2_rotated_cutout], dim=0)

        imgs_rotated = imgs_rotated.to(self.device, non_blocking=True)
        rotations = rotations.to(self.device, non_blocking=True)
        imgs_rotated_cutout = imgs_rotated_cutout.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            rotations_pred, contrast_pred, reconstructions = self.network(imgs_rotated_cutout)
            l = self.loss(rotations_pred, rotations, contrast_pred, reconstructions, imgs_rotated)

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

        imgs_rotated = torch.cat([imgs1_rotated, imgs2_rotated], dim=0)
        rotations = torch.cat([rotations1, rotations2], dim=0)
        imgs_rotated_cutout = torch.cat([imgs1_rotated_cutout, imgs2_rotated_cutout], dim=0)

        imgs_rotated = imgs_rotated.to(self.device, non_blocking=True)
        rotations = rotations.to(self.device, non_blocking=True)
        imgs_rotated_cutout = imgs_rotated_cutout.to(self.device, non_blocking=True)

        with torch.no_grad():
            with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                rotations_pred, contrast_pred, reconstructions = self.network(imgs_rotated_cutout)
                l = self.loss(rotations_pred, rotations, contrast_pred, reconstructions, imgs_rotated)

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


####################################################################
############################# VARIANTS #############################
####################################################################


class SwinUNETRTrainer_BS2(SwinUNETRTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 2


class SwinUNETRTrainer_BS8(SwinUNETRTrainer):

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

                l = self.loss(
                    rotations_pred, rotations, contrast1_pred, contrast2_pred, reconstructions, imgs_rotated
                )

        return {"loss": l.detach().cpu().numpy()}
