from typing import Union, Tuple, List

import numpy as np
import torch
from torch import nn
from torch.optim.adamw import AdamW
from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from torch import autocast
from nnssl.architectures.build_architecture import build_network_architecture
from nnssl.architectures.voco_architecture import VoCoArchitecture
from nnssl.training.loss.contrastive_loss import NTXentLoss
from nnssl.utilities.helpers import dummy_context


from einops import rearrange


from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.ssl_data.configure_basic_dummyDA import (
    configure_rotation_dummyDA_mirroring_and_inital_patch_size,
)
from nnssl.ssl_data.dataloading.voco_transform import VocoTransform
from nnssl.ssl_data.limited_len_wrapper import LimitedLenWrapper


from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer

from nnssl.utilities.default_n_proc_DA import get_allowed_n_proc_DA


class SimCLRTrainer(AbstractBaseTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        # Volume Contrastive uses this patch size, but prolly because of the crops
        # plan.configurations[configuration_name].patch_size = (192, 192, 64)
        super().__init__(
            plan, configuration_name, fold, dataset_json, unpack_dataset, device
        )
        self.batch_size = plan.configurations[configuration_name].batch_size

        self.initial_lr = 1e-3
        self.weight_decay = 1e-2

        # VoCo vars for debugging:
        patch_size = self.config_plan.patch_size
        self.voco_base_crop_count = (3, 3, 1)
        self.voco_target_crop_count = (
            5  # Number of crops to sample from each image.  Originally 4.
        )
        self.pred_loss_weight = 1
        self.reg_loss_weight = 1
        self.voco_crop_size = (
            patch_size[0] // self.voco_base_crop_count[0],
            patch_size[1] // self.voco_base_crop_count[1],
            patch_size[2] // self.voco_base_crop_count[2],
        )

    def configure_optimizers(self):
        optimizer = AdamW(
            params=self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
        )
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=10,
            max_epochs=self.num_epochs,
            warmup_start_lr=self.initial_lr / 100,
            eta_min=1e-6,
        )
        return optimizer, lr_scheduler

    def build_loss(self) -> nn.Module:
        """Implements the standard contrastive loss."""
        return NTXentLoss(
            batch_size=self.batch_size,
            temperature=0.5,
            similarity_function="cosine",
            device=self.device,
        )

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
            raise NotImplementedError(
                "We don't do dummy 2d aug here anymore. Data should be isotropic!"
            )

        patch_size_spatial = patch_size
        ignore_axes = None

        # tr_transforms.append(
        #     SpatialTransform(
        #         patch_size_spatial,
        #         patch_center_dist_from_border=None,
        #         do_elastic_deform=False,
        #         alpha=(0, 0),
        #         sigma=(0, 0),
        #         do_rotation=True,
        #         angle_x=rotation_for_DA["x"],
        #         angle_y=rotation_for_DA["y"],
        #         angle_z=rotation_for_DA["z"],
        #         p_rot_per_axis=1,  # todo experiment with this
        #         do_scale=True,
        #         scale=(0.7, 1.4),
        #         border_mode_data="constant",
        #         border_cval_data=0,
        #         order_data=order_resampling_data,
        #         border_mode_seg="constant",
        #         border_cval_seg=border_val_seg,
        #         order_seg=order_resampling_seg,
        #         random_crop=False,  # random cropping is part of our dataloaders
        #         p_el_per_sample=0,
        #         p_scale_per_sample=0.2,
        #         p_rot_per_sample=0.2,
        #         independent_scale_for_each_axis=False,  # todo experiment with this
        #     )
        # )

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

        tr_transforms.append(
            NumpyToTensor(["all_crops", "base_target_crop_overlaps"], "float")
        )
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

        val_transforms.append(
            NumpyToTensor(["all_crops", "base_target_crop_overlaps"], "float")
        )
        val_transforms = Compose(val_transforms)
        return val_transforms

    def get_dataloaders(self):
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.config_plan.patch_size
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
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

        # ----------------------- Validation data augmentations ---------------------- #
        val_transforms = self.get_validation_transforms()

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

    def build_architecture(
        self,
        config_plan: ConfigurationPlan,
        num_input_channels: int,
        num_output_channels: int,
    ) -> nn.Module:
        encoder = build_network_architecture(
            config_plan,
            num_input_channels,
            num_output_channels,
            encoder_only=True,
        )
        # Turns out VoCoArchitecture can be used for SimCLR purpose here.
        architecture = VoCoArchitecture(encoder, config_plan)
        return architecture

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
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            emeddings = self.network(all_crops)
            base_embeddings = rearrange(
                emeddings[:NBASE], "(b NBASE) c -> b NBASE c", b=self.batch_size
            )
            target_embeddings = rearrange(
                emeddings[NBASE:], "(b nTARGET) c -> b nTARGET c", b=self.batch_size
            )

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
            with (
                autocast(self.device.type, enabled=True)
                if self.device.type == "cuda"
                else dummy_context()
            ):
                emeddings = self.network(all_crops)
                base_embeddings = emeddings[:NBASE]
                target_embeddings = emeddings[NBASE:]
                base_embeddings = rearrange(
                    emeddings[:NBASE], "(b NBASE) c -> b NBASE c ", b=self.batch_size
                )
                target_embeddings = rearrange(
                    emeddings[NBASE:], "(b nTARGET) c -> b nTARGET c", b=self.batch_size
                )

                # del data
                l = self.loss(base_embeddings, target_embeddings, gt_overlaps)

        return {"loss": l.detach().cpu().numpy()}
