import torch
from math import prod
from torch import nn
from typing_extensions import override

from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.ssl_data.dataloading.pcrlv2_transform import PCRLv2Transform
from nnssl.ssl_data.dataloading.swin_unetr_transform import SwinUNETRTransform

from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer
from nnssl.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnssl.ssl_data.limited_len_wrapper import LimitedLenWrapper
from torch import autocast
from nnssl.utilities.helpers import dummy_context

from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor



class PCLRv2Trainer(AbstractBaseTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
        global_patch_sizes: tuple[tuple[int, int, int]] = ((64, 64, 32), (96, 96, 64), (96, 96, 96), (112, 112, 64)),
        global_input_size: tuple[int, int, int] = (64, 64, 32),
        local_patch_sizes: tuple[tuple[int, int, int]] = ((8, 8, 8), (16, 16, 16), (32, 32, 16), (32, 32, 32)),
        local_input_size: tuple[int, int, int] = (16, 16, 16),
        num_locals: int = 6,
        min_IoU: float = 0.3
    ):
        self._check_global_ps_and_IoU_validity(global_patch_sizes, min_IoU)

        # We want the dataloader to give us a patch_size big enough, to accommodate for the largest patch size
        # for each axis
        patch_size = tuple([2*max(patch_side_lengths) for patch_side_lengths in zip(global_patch_sizes)])
        plan.configurations[configuration_name].patch_size = patch_size

        super().__init__(plan, configuration_name, fold, pretrain_json, device)

        self.global_patch_sizes = global_patch_sizes
        self.global_input_size = global_input_size
        self.local_patch_sizes = local_patch_sizes
        self.local_input_size = local_input_size
        self.num_locals = num_locals
        self.min_IoU = min_IoU


    @staticmethod
    def _check_global_ps_and_IoU_validity(global_patch_sizes, min_IoU) -> None:
        """
        Make sure that the IoU threshold can be reached by the smallest and largest patch provided by
        'global_patch_sizes', otherwise we would run into an impossible task.
        """
        global_volumes = [prod(patch) for patch in global_patch_sizes]
        smallest_patch_size = global_patch_sizes[global_volumes.index(min(global_volumes))]
        largest_patch_size = global_patch_sizes[global_volumes.index(max(global_volumes))]
        maximum_intersect = prod([min(s, l) for s, l in zip(smallest_patch_size, largest_patch_size)])
        if maximum_intersect < min_IoU:
            raise ValueError(
                f"The maximum possible intersection ({maximum_intersect}) of {smallest_patch_size} and "
                f"{global_patch_sizes} is smaller than the minimum required IoU ({min_IoU}). "
                "Adjust the global patch sizes or lower the IoU threshold."
            )

    @override
    def build_loss(self):
        return PCLRv2Loss()

    @override
    def build_architecture(
        self, config_plan: ConfigurationPlan, num_input_channels: int, num_output_channels: int
    ) -> nn.Module:
        network = None # need to get rid of skip connections
        architecture = PCLRv2Architecture(network, num_input_channels)
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

        tr_transforms.append(PCRLv2Transform())
        tr_transforms.append(NumpyToTensor(cast_to="float", keys=["imgs_rotated", "imgs_rotated_cutout"]))
        tr_transforms.append(NumpyToTensor(cast_to="long", keys="rotations"))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms

    @staticmethod
    def get_validation_transforms() -> AbstractTransform:
        return PCLRv2Trainer.get_training_transforms()



