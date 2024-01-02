from abc import ABC, abstractmethod
from nnssl.architectures.build_architecture import build_network_architecture
from nnssl.architectures.convert_to_spark import convert_to_spark_cnn
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan
from nnssl.ssl_data.configure_basic_dummyDA import configure_rotation_dummyDA_mirroring_and_inital_patch_size
from nnssl.ssl_data.limited_len_wrapper import LimitedLenWrapper
from nnssl.training.nnsslTrainer.nnsslTrainer import AbstractnnsslTrainer
from nnssl.training.loss.spark_loss import SparkLoss
from torch import nn
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

from nnssl.utilities.default_n_proc_DA import get_allowed_n_proc_DA


class nnAbstractMAETrainer(AbstractnnsslTrainer, ABC):
    def _build_loss(self):
        """
        This is where you build your loss function. You can use anything from torch.nn here
        :return:
        """
        # return nn.MSELoss()
        return SparkLoss()

    def build_architecture(
        self, config_plan: ConfigurationPlan, num_input_channels: int, num_output_channels: int
    ) -> nn.Module:
        architecture = build_network_architecture(config_plan, num_input_channels, num_output_channels)
        spark_architecture = convert_to_spark_cnn(architecture)
        return spark_architecture

    def get_dataloaders(self):
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.config_plan.patch_size
        dim = len(patch_size)
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

        dl_tr, dl_val = self.get_plain_dataloaders(initial_patch_size)

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
