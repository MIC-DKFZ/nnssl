import torch
from torch import nn
from nnssl.architectures.spark_model import SparK3D
from nnssl.architectures.spark_utils import convert_to_spark_cnn
from nnssl.experiment_planning.experiment_planners.plan import Plan
from nnssl.training.lr_scheduler.polylr import ContinuedPolyLRSchedulerWithWarmup
from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import create_blocky_mask
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from nnssl.training.nnsslTrainer.masked_image_modeling.SparkTrainer import SparkMAETrainer
import numpy as np


class BaseVariableSparkMAETrainer(SparkMAETrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_json, pretrain_dataset, device)
        self.mask_percentage = (0.6, 0.9)
        self.num_epochs = 52
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

    def build_architecture(self, *args, **kwargs) -> nn.Module:
        n_stages = 6
        network = ResidualEncoderUNet(
            input_channels=1,
            n_stages=n_stages,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
            num_classes=1,
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=False,
        )

        spark_architecture = convert_to_spark_cnn(network.encoder)
        network.encoder = spark_architecture
        actual_network = SparK3D(network, (160, 160, 160))

        return actual_network


class VariableSparkMAETrainer_5ep(BaseVariableSparkMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 6  # 6 * 8 (8 GPUs)
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset, device)
        self.num_epochs = 5


class VariableSparkMAETrainer_BS6_ep1000(BaseVariableSparkMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 6  # 6 * 8 (8 GPUs)
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset, device)
        self.num_epochs = 1000


class BigVariableSparkMAETrainer(BaseVariableSparkMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 48  # 6 * 8 (8 GPUs)
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset, device)
        self.num_epochs = 4000
        self.initial_lr = 3e-2  # Bit more as we increase batch size a lot


class BigVariableSparkMAETrainerContinue(BigVariableSparkMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset, device)
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
