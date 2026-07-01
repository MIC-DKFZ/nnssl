from typing import Tuple, Union, override
import numpy as np
from torch.nn.modules import Module
from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans
from nnssl.architectures.get_network_by_name import get_network_by_name
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
import torch
from torch import autocast, distributed as dist

from nnssl.ssl_data.data_augmentation.transforms_for_dummy_2d import Convert3DTo2DTransform
from nnssl.ssl_data.dataloading.data_loader_3d import nnsslDataLoader3D, nnsslCenterCropDataLoader3D
from nnssl.ssl_data.dataloading.volume_fusion_transform import VolumeFusionTransform
from nnssl.training.loss.compound_losses import DC_and_CE_loss
from nnssl.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer

from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform,
    ContrastAugmentationTransform,
    GammaTransform,
)
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import save_json
from nnssl.ssl_data.limited_len_wrapper import LimitedLenWrapper
from nnssl.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnssl.utilities.helpers import dummy_context


class VolumeFusionTrainer(AbstractBaseTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
        foreground_classes: int = 5,
    ):
        # We increase the batch size by 2x because we mix the two samples together!
        """
        Paper hyperparameter details:
        1. Batch Size: 4
        2. Adam optimizer (but they also have a Hybrid CNN+Trans architecture)
        3. Init. Learning rate 10e-3
        4. Weight decay 10e-5

        They have multiple (data) scales:
        1. PData-1k
            1. 100k iterations
            2. Halved lr every 20k iterations
        2. PData-10k
            1. 200k iterations
            2. Halved lr every 40k iterations
        3. PData-110k
            1. 400k iterations
            2. Halved lr every 50k iterations

        Training conducted on 2 RTX3090 GPUs

        Downstream (finetune strategy):
        - Batch size 2
        - Init. LR 10e-3
        - LR halved every 5k iterations
        """

        # -------------------------------- DISCLAIMER -------------------------------- #
        # We don't set the hyperparameters as they do in the paper. Instead, we use the
        # hyperparameters from the nnUNet paper. This allows us to compare the methods
        # on a more equal footing. The hyperparameters from the paper are commented above.

        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        """
        Mis-FM predicts a (local) mixing factor between two images.
        This mixing factor is composed of boxes, with different factors 0/0.25/0.5/0.75/1.0
        This factor is local, so we do a dense prediction of this factor for each pixel.
        Hence, it is very similar to segmentation but slighly adapted loss function.

        Funnily enough they actually do classification, and don't consider the ordinality of the classes.
        This penalizes (IMO) confusions of 0.25 to 0.5 the same as 0.25 to 0.75 -- which is more "wrong".

        Details for reproduction from (https://arxiv.org/pdf/2306.16925) the paper:
        1. Split batch into two halves, one being "background" and one being "foreground"
        2. Create the new image through mixing patches X = αI_f + (1 − α)I_b
            1. α is the mixing factor; I_f is the foreground image ; I_b is the background image
            2. α is drawn from the set of V ={0.0, 1/K, 2/K, ...,(K − 1)/K, 1.0}
            3. K is the number of classes (not really the number of non-zero fusion coefficients or math doesn't checkout -- Paper has bad math)
            4. Important: Foreground patches are randomly and sequentially generated at various scales and aspect ratios,
                and the region for each class can have irregular shapes due to overlapping of the sequential patches.
            5. Patches -- with original patch size (64, 128, 128):
                1. M is U(10, 40)
                2. Depth U(8, 40) -> [12,5%, 62.5%] of the original depth
                3. height U(8, 80) -> [6.25%, 62.5%] of the original height
                4. width U(8, 80) -> [6.25%, 62.5%] of the original width
                # We use the range of [6.25%, 62.5%] of the original size as range! (not the absolute values)
            6. Number of classes K is 4.
        """
        # MisFM specific parameters:
        self.config_plan.patch_size = (160, 160, 160)
        self.num_output_channels = foreground_classes
        # vf == volume fusion
        self.vf_mixing_coefficients = np.linspace(
            0, 1, foreground_classes, endpoint=True
        )  # [0, 0.25, 0.5, 0.75, 1.0] if foreground_classes=5
        self.vf_subpatch_count = (10, 40)  # U(10, 40) upper bound is excluded in the range

        # Max depth, height, width patch scale.
        self.vf_subpatch_size = [(8, int(0.625 * s) + 1) for s in self.config_plan.patch_size]

    @override
    def build_architecture_and_adaptation_plan(
        self, config_plan: ConfigurationPlan, num_input_channels: int, num_output_channels: int, *args, **kwargs
    ) -> Module:
        architecture = get_network_by_name(
            config_plan,
            "ResEncL",
            num_input_channels,
            num_output_channels,
        )
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

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...]):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        # We double the batch size because we mix two samples together
        #   Training sample batch size is /2, hence the plan refers to the batch size of samples fed to the network.
        load_batchsize = self.batch_size * 2

        dl_tr = nnsslDataLoader3D(
            dataset_tr,
            load_batchsize,  # We double the batch size because we mix two samples together
            initial_patch_size,
            self.config_plan.patch_size,
            sampling_probabilities=None,
            pad_sides=None,
        )
        dl_val = nnsslDataLoader3D(
            dataset_val,
            load_batchsize,  # We double the batch size because we mix two samples together
            self.config_plan.patch_size,
            self.config_plan.patch_size,
            sampling_probabilities=None,
            pad_sides=None,
        )
        return dl_tr, dl_val

    def get_centercrop_dataloaders_with_doubled_batch_size(self):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = nnsslCenterCropDataLoader3D(
            dataset_tr,
            2 * self.batch_size,
            self.config_plan.patch_size,
            self.config_plan.patch_size,
            sampling_probabilities=None,
            pad_sides=None,
        )
        dl_val = nnsslCenterCropDataLoader3D(
            dataset_val,
            2 * self.batch_size,
            self.config_plan.patch_size,
            self.config_plan.patch_size,
            sampling_probabilities=None,
            pad_sides=None,
        )
        return dl_tr, dl_val

    def build_loss(self):
        loss = DC_and_CE_loss(
            {
                "batch_dice": False,
                "smooth": 1e-5,
                "do_bg": True,  # There is no real "background" class.
                "ddp": self.is_ddp,
            },
            {},
            weight_ce=1,
            weight_dice=1,
            ignore_label=None,
            dice_class=MemoryEfficientSoftDiceLoss,
        )
        return loss

    def train_step(self, batch: dict) -> dict:
        data = batch["input"]
        label = batch["target"]
        data = data.to(self.device, non_blocking=True)
        label = label.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            # del data
            l = self.loss(output, label)

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
        data = batch["input"]
        label = batch["target"]
        data = data.to(self.device, non_blocking=True)
        label = label.to(self.device, non_blocking=True)

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with torch.no_grad():
            with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                output = self.network(data)
                # del data
                l = self.loss(output, label)

        return {"loss": l.detach().cpu().numpy()}

    def get_dataloaders(self):
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.config_plan.patch_size

        # ------------------------ Training data augmentations ----------------------- #
        tr_transforms = self.get_training_transforms(
            patch_size=patch_size,
            rotation_for_DA=None,
            mirror_axes=(0, 1, 2),
            do_dummy_2d_data_aug=False,
            order_resampling_data=3,
            order_resampling_seg=1,
        )

        # ----------------------- Validation data augmentations ---------------------- #
        val_transforms = self.get_validation_transforms()

        dl_tr, dl_val = self.get_centercrop_dataloaders_with_doubled_batch_size()
        # dl_tr, dl_val = self.get_plain_dataloaders(patch_size)

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
            raise NotImplementedError("No dummy 2D data augmentation support atm.")

        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
        tr_transforms.append(
            GaussianBlurTransform((0.5, 1.0), different_sigma_per_channel=True, p_per_sample=0.2, p_per_channel=0.5)
        )
        tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
        tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
        tr_transforms.append(
            SimulateLowResolutionTransform(
                zoom_range=(0.5, 1),
                per_channel=True,
                p_per_channel=0.5,
                order_downsample=0,
                order_upsample=3,
                p_per_sample=0.25,
                ignore_axes=None,
            )
        )
        tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
        tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

        if mirror_axes is not None and len(mirror_axes) > 0:
            tr_transforms.append(MirrorTransform(mirror_axes))

        # -------- This is the important trafo or the Volume Fusion Transform -------- #
        tr_transforms.append(
            VolumeFusionTransform(self.vf_mixing_coefficients, self.vf_subpatch_count, self.vf_subpatch_size)
        )
        tr_transforms.append(NumpyToTensor(["input", "target"], "float"))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms

    def get_validation_transforms(self) -> AbstractTransform:
        val_transforms = []

        # -------- This is the important trafo or the Volume Fusion Transform -------- #
        val_transforms.append(
            VolumeFusionTransform(self.vf_mixing_coefficients, self.vf_subpatch_count, self.vf_subpatch_size)
        )
        val_transforms.append(NumpyToTensor(["input", "target"], "float"))
        val_transforms = Compose(val_transforms)
        return val_transforms


####################################################################
############################# VARIANTS #############################
####################################################################


class VolumeFusionTrainer_test(VolumeFusionTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].patch_size = (96, 96, 96)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 4


class VolumeFusionTrainer_BS8(VolumeFusionTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].patch_size = (160, 160, 160)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 8


############################# LEARNING RATE #############################


class VolumeFusionTrainer_BS8_lr_1e3(VolumeFusionTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].patch_size = (160, 160, 160)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 8
        self.initial_lr = 1e-3


class VolumeFusionTrainer_BS8_lr_1e4(VolumeFusionTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].patch_size = (160, 160, 160)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 8
        self.initial_lr = 1e-4


############################# WEIGHT DECAY #############################


class VolumeFusionTrainer_BS8_lr_1e3_wd_3e4(VolumeFusionTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].patch_size = (160, 160, 160)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 8
        self.initial_lr = 1e-3
        self.weight_decay = 3e-4


class VolumeFusionTrainer_BS8_lr_1e3_wd_3e6(VolumeFusionTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].patch_size = (160, 160, 160)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 8
        self.initial_lr = 1e-3
        self.weight_decay = 3e-6


############################# FOREGROUND CLASSES #############################


class VolumeFusionTrainer_BS8_lr_1e3_wd_3e5_C3(VolumeFusionTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].patch_size = (160, 160, 160)
        super().__init__(plan, configuration_name, fold, pretrain_json, device, foreground_classes=3)
        self.total_batch_size = 8
        self.initial_lr = 1e-3
        self.weight_decay = 3e-5


class VolumeFusionTrainer_BS8_lr_1e3_wd_3e5_C9(VolumeFusionTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].patch_size = (160, 160, 160)
        super().__init__(plan, configuration_name, fold, pretrain_json, device, foreground_classes=9)
        self.total_batch_size = 8
        self.initial_lr = 1e-3
        self.weight_decay = 3e-5
