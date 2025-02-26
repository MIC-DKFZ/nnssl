import random
from itertools import combinations

import torch
from math import prod
from torch import nn
from typing_extensions import override

from nnssl.architectures.nsUNet import ResidualEncoderUNet_noskip
from nnssl.architectures.pclrv2_architecture import PCLRv2Architecture
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.ssl_data.dataloading.pcrlv2_transform import PCRLv2Transform, Shape3D
from nnssl.ssl_data.dataloading.swin_unetr_transform import SwinUNETRTransform
from nnssl.training.loss.pclrv2_loss import PCLRv2Loss

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
        # my config
        global_patch_sizes: tuple[Shape3D] = ((96, 96, 96), (128, 128, 96), (128, 128, 128), (160, 160, 128)),
        global_input_size: Shape3D = (128, 128, 128),
        local_patch_sizes: tuple[Shape3D] = ((32, 32, 32), (64, 64, 32), (64, 64, 64)),
        local_input_size: Shape3D = (64, 64, 64),
        # orig config
        # global_patch_sizes: tuple[Shape3D] = ((64, 64, 32), (96, 96, 64), (96, 96, 96), (112, 112, 64)),
        # global_input_size: Shape3D = (64, 64, 32),
        # local_patch_sizes: tuple[Shape3D] = ((8, 8, 8), (16, 16, 16), (32, 32, 16), (32, 32, 32)),
        # local_input_size: Shape3D = (16, 16, 16),
        num_locals: int = 6,
        min_IoU: float = 0.3
    ):
        self._check_global_ps_and_IoU_validity(global_patch_sizes, min_IoU)

        # We want the dataloader to give us a patch_size big enough, to accommodate for the largest patch size
        # for each axis
        # patch_size = tuple([2*max(patch_side_lengths) for patch_side_lengths in zip(global_patch_sizes)])
        # plan.configurations[configuration_name].patch_size = patch_size
        plan.configurations[configuration_name].patch_size = (180, 180, 180)

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
        return PCLRv2Loss(self.num_mid_stages, self.num_locals)

    @override
    def build_architecture(
        self, config_plan: ConfigurationPlan, num_input_channels: int, num_output_channels: int
    ) -> nn.Module:
        network = ResidualEncoderUNet_noskip(
            input_channels=1,
            n_stages=6,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(6)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
            num_classes=1,
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True}
        )
        architecture = PCLRv2Architecture(network)
        self.num_mid_stages = len(architecture.features_per_mid_stage)
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

        aug_global_crops_A = batch["aug_global_crops_A"]    # [B,              C, X_global_input_size, Y_global_input_size, Z_global_input_size]
        global_crops_A = batch["global_crops_A"]            # [B,              C, X_global_input_size, Y_global_input_size, Z_global_input_size]
        aug_global_crops_B = batch["aug_global_crops_B"]    # [B,              C, X_global_input_size, Y_global_input_size, Z_global_input_size]
        aug_local_crops = batch["aug_local_crops"]          # [(B*num_locals), C, X_local_input_size,  Y_local_input_size,  Z_local_input_size ]

        aug_global_crops_A = aug_global_crops_A.to(self.device, non_blocking=True)
        global_crops_A = global_crops_A.to(self.device, non_blocking=True)
        aug_global_crops_B = aug_global_crops_B.to(self.device, non_blocking=True)
        aug_local_crops = aug_local_crops.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            reconstructions_A, embeddings_A, mid_reconstructions_A = self.network(aug_global_crops_A)
            embeddings_B = self.network(aug_global_crops_B, embeddings_only=True)
            local_embeddings = self.network(aug_local_crops, embeddings_only=True)
            l = self.loss(reconstructions_A, mid_reconstructions_A, global_crops_A, embeddings_A, embeddings_B,
                          local_embeddings)

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

    #TODO
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

    def get_training_transforms(self) -> AbstractTransform:
        tr_transforms = Compose([PCRLv2Transform(
            self.global_patch_sizes,
            self.global_input_size,
            self.local_patch_sizes,
            self.local_input_size,
            self.num_locals,
            self.min_IoU,
        )])
        return tr_transforms

    def get_validation_transforms(self) -> AbstractTransform:
        return self.get_training_transforms()


####################################################################
############################# VARIANTS #############################
####################################################################


class PCLRv2Trainer_test(PCLRv2Trainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,

        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 2
        super().__init__(plan, configuration_name, fold,  pretrain_json, device,
                        global_input_size=(64, 64, 64))
