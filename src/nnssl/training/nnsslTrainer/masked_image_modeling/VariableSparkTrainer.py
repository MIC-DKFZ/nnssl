from typing import override
import torch
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import save_json
from torch import nn, autocast

from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans
from nnssl.architectures import spark_utils
from nnssl.architectures.get_network_by_name import get_network_by_name
from nnssl.architectures.spark_model import SparK3D
from nnssl.architectures.spark_utils import convert_to_spark_cnn
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.ssl_data.configure_basic_dummyDA import configure_rotation_dummyDA_mirroring_and_inital_patch_size
from nnssl.ssl_data.limited_len_wrapper import LimitedLenWrapper
from nnssl.training.loss.mse_loss import LossMaskMSELoss
from nnssl.training.lr_scheduler.polylr import ContinuedPolyLRSchedulerWithWarmup
from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import create_blocky_mask
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from nnssl.training.nnsslTrainer.masked_image_modeling.SparkTrainer import SparkMAETrainer
import numpy as np

from nnssl.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnssl.utilities.helpers import dummy_context


class BaseVariableSparkMAETrainer(SparkMAETrainer):

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
        self.mask_percentage = (0.6, 0.9)
        # self.num_epochs = 52
        self.mask_random_seed = np.random.RandomState(123)

    def mask_creation(
        self,
        batch_size: int,
        patch_size: tuple[int, int, int],
        mask_percentage: tuple[float, float],
        rng_seed: int | None = None,
    ) -> torch.Tensor:
        """
        Creates a masking tensor with 1s (indicating no masking) and 0s (indicating masking).
        The mask has to be of same size like the input data (batch_size, 1, x, y, z).

        :param patch_shape: The 3D shape information for the masking patch.
        :param mask_percentage: percentage of the patch that should be masked
        :param min_mask_block_size: minimum size of the blocks that should be masked
        :return:
        """

        block_size = 16

        cur_mask_ratio = self.mask_random_seed.uniform(mask_percentage[0], mask_percentage[1])
        mask = [create_blocky_mask(patch_size, block_size, cur_mask_ratio) for _ in range(batch_size)]
        mask = torch.stack(mask)[:, None, ...]  # Add channel dimension
        return mask

    @override
    def build_architecture_and_adaptation_plan(
        self, config_plan: ConfigurationPlan, num_input_channels: int, num_output_channels: int
    ) -> nn.Module:
        network = get_network_by_name(
            config_plan,
            "ResEncL",
            num_input_channels,
            num_output_channels,
        )

        spark_architecture = convert_to_spark_cnn(network.encoder)
        network.encoder = spark_architecture

        actual_network = SparK3D(network)
        # ------------------------------ Adaptation Plan ----------------------------- #
        adapt_plan = AdaptationPlan(
            architecture_plans=ArchitecturePlans("ResEncL"),
            pretrain_plan=self.plan,
            pretrain_num_input_channels=1,
            recommended_downstream_patchsize=self.recommended_downstream_patchsize,
            key_to_encoder="encoder.stages",
            key_to_stem="encoder.stem",
            keys_to_in_proj=("encoder.stem.convs.0.conv", "encoder.stem.convs.0.all_modules.0"),
        )

        return actual_network, adapt_plan


class BaseVariableSparkMAETrainer_ANAT(BaseVariableSparkMAETrainer):
    def get_dataloaders(self):
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
            use_mask_for_norm=self.config_plan.use_mask_for_norm,
        )

        # ----------------------- Validation data augmentations ---------------------- #
        val_transforms = self.get_validation_transforms()

        dl_tr, dl_val = self.get_foreground_dataloaders(initial_patch_size)

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


class BaseVariableSparkMAETrainer_ANON(BaseVariableSparkMAETrainer):
    def build_loss(self):
        return LossMaskMSELoss()

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        anon = batch["seg"]
        data = data.to(self.device, non_blocking=True)
        anon = anon.to(self.device, non_blocking=True)

        mask = self.mask_creation(self.batch_size, self.config_plan.patch_size, self.mask_percentage).to(
            self.device, non_blocking=True
        )
        spark_utils._cur_active = mask

        # 'SparkLoss' scales the mask to the voxel space during the forward call.
        # However, since we want to apply the anonymization mask as well, we have to do all the calculations here,
        # just like BaseMAETrainer does
        rep_D, rep_H, rep_W = (
            data.shape[2] // mask.shape[2],
            data.shape[3] // mask.shape[3],
            data.shape[4] // mask.shape[4],
        )
        mask = mask.repeat_interleave(rep_D, dim=2).repeat_interleave(rep_H, dim=3).repeat_interleave(rep_W, dim=4)
        loss_mask = (1 - mask) * (1 - anon)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            l = self.loss(output, data, loss_mask)
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
        with torch.no_grad():
            data = batch["data"]
            anon = batch["seg"]
            data = data.to(self.device, non_blocking=True)
            anon = anon.to(self.device, non_blocking=True)

            mask = self.mask_creation(self.batch_size, self.config_plan.patch_size, self.mask_percentage).to(
                self.device, non_blocking=True
            )
            spark_utils._cur_active = mask

            rep_D, rep_H, rep_W = (
                data.shape[2] // mask.shape[2],
                data.shape[3] // mask.shape[3],
                data.shape[4] // mask.shape[4],
            )
            mask = (
                mask.repeat_interleave(rep_D, dim=2).repeat_interleave(rep_H, dim=3).repeat_interleave(rep_W, dim=4)
            )
            loss_mask = (1 - mask) * (1 - anon)

            with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                output = self.network(data)
            l = self.loss(output, data, loss_mask)
            return {"loss": l.detach().cpu().numpy()}


class BaseVariableSparkMAETrainer_ANAT_ANON(BaseVariableSparkMAETrainer_ANAT, BaseVariableSparkMAETrainer_ANON):
    pass


class VariableSparkMAETrainer_BS8(BaseVariableSparkMAETrainer):
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


class VariableSparkMAETrainer_ANAT_ANON_BS8(BaseVariableSparkMAETrainer_ANAT_ANON):
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


class VariableSparkMAETrainer_ANAT_ANON_test(BaseVariableSparkMAETrainer_ANAT_ANON):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 1


class VariableSparkMAETrainer_BS1(BaseVariableSparkMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 1


class VariableSparkMAETrainer_5ep(BaseVariableSparkMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 6
        self.num_epochs = 5


class VariableSparkMAETrainer_BS6_ep1000(BaseVariableSparkMAETrainer):
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
        self.num_epochs = 1000


class BigVariableSparkMAETrainer(BaseVariableSparkMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 48  # 6 * 8 (8 GPUs)
        self.num_epochs = 4000
        self.initial_lr = 3e-2  # Bit more as we increase batch size a lot


class BigVariableSparkMAETrainerContinue(BigVariableSparkMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.num_epochs = 5000  # Add 1k epochs to the previous.

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True
        )
        lr_scheduler = ContinuedPolyLRSchedulerWithWarmup(
            optimizer,
            start_epoch=self.current_epoch,
            initial_lr=7e-3,
            warmup_lr=1e-5,
            final_lr=1e-5,
            total_epochs=self.num_epochs,
            warmup_epochs=50,
        )
        return optimizer, lr_scheduler
