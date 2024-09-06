from typing import Union, Tuple, List

import numpy as np
import torch
from torch import nn
from torch.optim.adamw import AdamW
from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from torch import autocast
from nnssl.architectures.build_architecture import build_network_architecture
from nnssl.architectures.voco_architecture import VoCoArchitecture
from nnssl.training.loss.contrastive_loss import NTXentLoss
from nnssl.utilities.helpers import dummy_context

from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.ssl_data.configure_basic_dummyDA import (
    configure_rotation_dummyDA_mirroring_and_inital_patch_size,
)
from nnssl.ssl_data.limited_len_wrapper import LimitedLenWrapper

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnssl.ssl_data.dataloading.simclr_transform import SimCLRTransform
from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer

from nnssl.utilities.default_n_proc_DA import get_allowed_n_proc_DA


class SimCLRTrainer(AbstractBaseTrainer):
    """
    TODO:
    - implement data aug path for simclr [x]
        - check which standard transforms to keep [x] - went with default nnUNet transforms fow now
    - re-use VoCoArchitecture (seems like no change necessary here, double-check) [x]
    - implement train/val steps (loss returns loss, accuracy) -> maybe track acc. similar to pseudo dice in nnUNet [x] - not tracking yet
    - re-implement similar to VoCoTransform (need more sub-crops, and random crops in general) [ ]
    - maybe force partial overlaps between crops [ ]
    - clean up, test runs [ ]
    """

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        # Let's use the same patch size as VoCo

        plan.configurations[configuration_name].patch_size = (192, 192, 64)
        plan.configurations[configuration_name].batch_size = 4  # TODO: test larger bs

        super().__init__(
            plan, configuration_name, fold, dataset_json, unpack_dataset, device
        )
        self.batch_size = plan.configurations[configuration_name].batch_size
        self.num_crops_per_image = 1
        self.crop_size = (64, 64, 64)
        self.min_crop_overlap = 0.5

        self.initial_lr = 1e-3
        self.weight_decay = 1e-2

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

        # --------------------------- SimCLR Transformation --------------------------- #
        # All train augmentations are moved to the SimCLR Transform class.

        tr_transforms.append(
            SimCLRTransform(
                crop_size=self.crop_size,
                aug="train",
                crop_count_per_image=self.num_crops_per_image,
                min_overlap_ratio=self.min_crop_overlap,
                data_key="data",
            )
        )
        # From here on out we are working with reference and overlapping crops!

        tr_transforms.append(NumpyToTensor(["all_crops"], "float"))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms

    def get_validation_transforms(
        self,
        deep_supervision_scales: Union[List, Tuple, None],
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:

        default_validation_transforms = nnUNetTrainer.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded,
            foreground_labels,
            regions,
            ignore_label,
        )
        return SimCLRTransform(default_validation_transforms)

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
            None,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.config_plan.use_mask_for_norm,
            is_cascaded=False,
            foreground_labels=None,
            regions=None,
            ignore_label=None,
        )

        # ----------------------- Validation data augmentations ---------------------- #
        val_transforms = self.get_validation_transforms(
            None,
            is_cascaded=False,
            foreground_labels=None,
            regions=None,
            ignore_label=None,
        )

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

    def train_step(self, batch: Tuple[dict, dict]) -> dict:

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

            z_i = self.network(
                batch["image_i"].unsqueeze(1).to(self.device, non_blocking=True)
            )
            z_j = self.network(
                batch["image_j"].unsqueeze(1).to(self.device, non_blocking=True)
            )

            # Normalize prior to contrastive loss
            z_i = nn.functional.normalize(z_i, dim=1)
            z_j = nn.functional.normalize(z_j, dim=1)

            # del data
            l, acc = self.loss(z_i, z_j)

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

        print(f"Train loss: {l.detach().cpu().numpy()} - accuracy: {acc}")

        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:

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
                z_i = self.network(
                    batch["image_i"].unsqueeze(1).to(self.device, non_blocking=True)
                )
                z_j = self.network(
                    batch["image_j"].unsqueeze(1).to(self.device, non_blocking=True)
                )

                # Normalize prior to contrastive loss
                z_i = nn.functional.normalize(z_i, dim=1)
                z_j = nn.functional.normalize(z_j, dim=1)

                # del data
                l, acc = self.loss(z_i, z_j)
                print(f"Val loss: {l.detach().cpu().numpy()} - accuracy: {acc}")

        return {"loss": l.detach().cpu().numpy()}
