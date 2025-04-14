from copy import deepcopy
from typing import Union, Tuple, override

import numpy as np
import torch
from torch import nn
from torch.optim.adamw import AdamW
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor

from torch import autocast
from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans
from nnssl.architectures.get_network_by_name import get_network_by_name
from nnssl.architectures.voco_architecture import VoCoArchitecture
from nnssl.training.loss.voco_loss import VoCoLoss
from nnssl.utilities.helpers import dummy_context
from batchgenerators.utilities.file_and_folder_operations import save_json


from einops import rearrange


from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.ssl_data.configure_basic_dummyDA import configure_rotation_dummyDA_mirroring_and_inital_patch_size
from nnssl.ssl_data.dataloading.voco_transform import VocoTransform
from nnssl.ssl_data.limited_len_wrapper import LimitedLenWrapper


from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer

from nnssl.utilities.default_n_proc_DA import get_allowed_n_proc_DA


class VoCoTrainer(AbstractBaseTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
        patch_size: tuple = (256, 256, 64),
        base_crop_count: tuple = (4, 4, 1),
        target_crop_count: int = 4,
    ):
        # plan.configurations[configuration_name].patch_size = (192, 192, 64)
        plan.configurations[configuration_name].patch_size = patch_size
        # x y z crop counts -- Official code has this [4, 4, 1].
        # Original code has patch size 384x384x96
        #   This does not make sense for our BrainMRIs who have 1x1x1 [mm] spacing.
        #   The brain is not 38,4cm wide, so scans are smaller.
        #   Moreover, they first crop 96x96x96 patches from the data.
        #   Then they resample the crops to 64x64x64; -- Makes no sense.
        # So instead we directly make z 64, and x and y 192, which is a nice 3:1 ratio.
        # This results in crops that are still the same size of 64x64x64;

        # I wish I could make patch_size and crop sizes bigger, but VRAM goes kaboom.
        # self.voco_base_crop_count = (3, 3, 1)
        self.voco_base_crop_count = base_crop_count

        # BS1 == 6GB VRAM
        # --> 40GB VRAM fits BS8
        # plan.configurations[configuration_name].batch_size = 1
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        patch_size = self.config_plan.patch_size

        # self.initial_lr = 1e-3
        # self.weight_decay = 1e-2

        self.voco_target_crop_count = target_crop_count  # Number of crops to sample from each image.  Originally 4.
        self.pred_loss_weight = 1
        self.reg_loss_weight = 1
        # Size of crops in voxels.
        assert (
            patch_size[0] % self.voco_base_crop_count[0] == 0
        ), f"Patch size must be divisible by crop count. {patch_size[0]} % {self.voco_base_crop_count[0]} = {patch_size[0] % self.voco_base_crop_count[0]}"
        assert (
            patch_size[1] % self.voco_base_crop_count[1] == 0
        ), f"Patch size must be divisible by crop count. {patch_size[1]} % {self.voco_base_crop_count[1]} = {patch_size[1] % self.voco_base_crop_count[1]}"
        assert (
            patch_size[2] % self.voco_base_crop_count[2] == 0
        ), f"Patch size must be divisible by crop count. {patch_size[2]} % {self.voco_base_crop_count[2]} = {patch_size[2] % self.voco_base_crop_count[2]}"

        self.voco_crop_size = (
            patch_size[0] // self.voco_base_crop_count[0],
            patch_size[1] // self.voco_base_crop_count[1],
            patch_size[2] // self.voco_base_crop_count[2],
        )

    # def configure_optimizers(self):
    #     optimizer = AdamW(
    #         params=self.network.parameters(),
    #         lr=self.initial_lr,
    #         weight_decay=self.weight_decay,
    #     )
    #     lr_scheduler = LinearWarmupCosineAnnealingLR(
    #         optimizer=optimizer,
    #         warmup_epochs=10,
    #         max_epochs=self.num_epochs,
    #         warmup_start_lr=self.initial_lr / 100,
    #         eta_min=1e-6,
    #     )
    #     return optimizer, lr_scheduler

    def build_loss(self) -> nn.Module:
        """Implements the VoCo loss, which forces rep similarity to be proportional to the volumetric overlap and for non-overlapping base crops to be orthogonal."""
        return VoCoLoss(pred_weight=self.pred_loss_weight, reg_weight=self.reg_loss_weight)

    def get_training_transforms(
        self,
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: dict,
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        order_resampling_data: int = 3,
        order_resampling_seg: int = 1,
        border_val_seg: int = -1,
    ) -> AbstractTransform:
        tr_transforms = []

        if do_dummy_2d_data_aug:
            raise NotImplementedError("We don't do dummy 2d aug here anymore. Data should be isotropic!")

        # --------------------------- VoCo Transformation --------------------------- #
        # All train augmentations are moved to the VoCoTransform class.
        #   This should help the crops to be more variable and hopefully makes the network better.
        tr_transforms.append(
            VocoTransform(
                voco_base_crop_count=self.voco_base_crop_count,
                voco_crop_size=self.voco_crop_size,
                aug="train",
                voco_target_crop_count=self.voco_target_crop_count,
                data_key="data",
            )
        )
        # From here on out we are working with base crops and target crops!

        tr_transforms.append(NumpyToTensor(["all_crops", "base_target_crop_overlaps"], "float"))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms

    def get_validation_transforms(self) -> AbstractTransform:
        val_transforms = []

        # --------------------------- VoCo Transformation --------------------------- #
        val_transforms.append(
            VocoTransform(
                voco_base_crop_count=self.voco_base_crop_count,
                voco_crop_size=self.voco_crop_size,
                aug="none",
                voco_target_crop_count=self.voco_target_crop_count,
                data_key="data",
            )
        )

        val_transforms.append(NumpyToTensor(["all_crops", "base_target_crop_overlaps"], "float"))
        val_transforms = Compose(val_transforms)
        return val_transforms

    def get_dataloaders(self):
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.config_plan.patch_size
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            _,
            mirror_axes,
        ) = configure_rotation_dummyDA_mirroring_and_inital_patch_size(patch_size)
        if do_dummy_2d_data_aug:
            self.print_to_log_file("Using dummy 2D data augmentation")

        # ------------------------ Training data augmentations ----------------------- #
        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            mirror_axes,
            do_dummy_2d_data_aug,
            order_resampling_data=3,
            order_resampling_seg=1,
        )
        val_transforms = self.get_validation_transforms()
        # ----------------------- Validation data augmentations ---------------------- #

        # We don't do non-90 degree rotations for the VoCo Trainer.
        dl_tr, dl_val = self.get_plain_dataloaders(patch_size)

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
        architecture = VoCoArchitecture(encoder, encoder.output_channels)

        # We need to set the patch size to the one the model saw during training
        plan = deepcopy(self.plan)
        plan.configurations[self.configuration_name].patch_size = self.voco_crop_size

        adapt_plan = AdaptationPlan(
            architecture_plans=ArchitecturePlans("ResEncL"),
            pretrain_plan=plan,
            recommended_downstream_patchsize=self.recommended_downstream_patchsize,
            pretrain_num_input_channels=num_input_channels,
            key_to_encoder="encoder.stages",
            key_to_stem="encoder.stem",
            keys_to_in_proj=("encoder.stem.convs.0.conv", "encoder.stem.convs.0.all_modules.0"),
        )
        return architecture, adapt_plan

    def train_step(self, batch: dict) -> dict:
        all_crops = batch["all_crops"]
        NBASE = batch["base_crop_index"]
        gt_overlaps = batch["base_target_crop_overlaps"]

        all_crops = all_crops.to(self.device, non_blocking=True)
        gt_overlaps = gt_overlaps.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            embeddings = self.network(all_crops)
            base_embeddings = rearrange(embeddings[:NBASE], "(b NBASE) c -> b NBASE c", b=self.batch_size)
            target_embeddings = rearrange(embeddings[NBASE:], "(b nTARGET) c -> b nTARGET c", b=self.batch_size)

            # del data
            l = self.loss(base_embeddings, target_embeddings, gt_overlaps)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            # torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            # torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        all_crops = batch["all_crops"]
        NBASE = batch["base_crop_index"]
        gt_overlaps = batch["base_target_crop_overlaps"]

        all_crops = all_crops.to(self.device, non_blocking=True)
        gt_overlaps = gt_overlaps.to(self.device, non_blocking=True)

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with torch.no_grad():
            with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                embeddings = self.network(all_crops)
                # base_embeddings = embeddings[:NBASE]
                # target_embeddings = embeddings[NBASE:]
                base_embeddings = rearrange(embeddings[:NBASE], "(b NBASE) c -> b NBASE c ", b=self.batch_size)
                target_embeddings = rearrange(embeddings[NBASE:], "(b nTARGET) c -> b nTARGET c", b=self.batch_size)

                # del data
                l = self.loss(base_embeddings, target_embeddings, gt_overlaps)

        return {"loss": l.detach().cpu().numpy()}


####################################################################
############################# VARIANTS #############################
####################################################################


class VoCoTrainer_test(VoCoTrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
            patch_size=(128, 128, 64),
            base_crop_count=(2, 2, 1),
        )
        self.total_batch_size = 1


############################# LEARNING RATE #############################


class VoCoTrainer_BS8_lr_1e2(VoCoTrainer):
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
        self.initial_lr = 1e-2


class VoCoTrainer_BS8_lr_1e3(VoCoTrainer):
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
        self.initial_lr = 1e-3


class VoCoTrainer_BS8_lr_1e4(VoCoTrainer):
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
        self.initial_lr = 1e-4


############################# WEIGHT DECAY #############################


class VoCoTrainer_BS8_lr_1e2_wd_3e4(VoCoTrainer):
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
        self.initial_lr = 1e-2
        self.weight_decay = 3e-4


class VoCoTrainer_BS8_lr_1e2_wd_3e6(VoCoTrainer):
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
        self.initial_lr = 1e-2
        self.weight_decay = 3e-6


class VoCoTrainer_BS8_lr_1e2_wd_3e2(VoCoTrainer):
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
        self.initial_lr = 1e-2
        self.weight_decay = 3e-2


############################# BASES & PATCH SIZE #############################


class VoCoTrainer_BS8_lr_1e2_wd_3e5_2x2x1_PS96(VoCoTrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
            patch_size=(192, 192, 96),
            base_crop_count=(2, 2, 1),
        )
        self.total_batch_size = 8


class VoCoTrainer_BS8_lr_1e2_wd_3e5_2x2x2_PS96(VoCoTrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
            patch_size=(192, 192, 192),
            base_crop_count=(2, 2, 2),
        )
        self.total_batch_size = 8


class VoCoTrainer_BS8_lr_1e2_wd_3e5_3x3x1_PS64(VoCoTrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
            patch_size=(192, 192, 64),
            base_crop_count=(3, 3, 1),
        )
        self.total_batch_size = 8


class VoCoTrainer_BS8_lr_1e2_wd_3e5_3x3x2_PS64(VoCoTrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
            patch_size=(192, 192, 128),
            base_crop_count=(3, 3, 2),
        )
        self.total_batch_size = 8


class VoCoTrainer_BS8_lr_1e2_wd_3e5_4x4x2_PS64(VoCoTrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
            patch_size=(256, 256, 128),
            base_crop_count=(4, 4, 2),
        )
        self.total_batch_size = 8


############################# NUMBER OF TARGET CROPS #############################


class VoCoTrainer_BS8_lr_1e2_wd_3e5_4x4x1_PS64_N2(VoCoTrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
            patch_size=(256, 256, 64),
            base_crop_count=(4, 4, 1),
            target_crop_count=2,
        )
        self.total_batch_size = 8


class VoCoTrainer_BS8_lr_1e2_wd_3e5_4x4x1_PS64_N8(VoCoTrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
            patch_size=(256, 256, 64),
            base_crop_count=(4, 4, 1),
            target_crop_count=8,
        )
        self.total_batch_size = 8
