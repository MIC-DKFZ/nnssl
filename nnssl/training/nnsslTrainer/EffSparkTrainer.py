import torch
from torch import nn
from nnssl.architectures.spark_model import EfficientSpark3D
from nnssl.architectures.spark_utils import convert_to_spark_cnn
from nnssl.experiment_planning.experiment_planners.plan import Plan
from nnssl.training.loss.spark_loss import SparkLoss
from nnssl.training.nnsslTrainer.SparkTrainer import SparkMAETrainer
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet


class EffSparkMAETrainer(SparkMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, dataset_json, unpack_dataset, device)
        self.network: EfficientSpark3D = ...

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
        actual_network = EfficientSpark3D(network, (160, 160, 160))

        return actual_network


class EffSparkMAETrainer_5ep(EffSparkMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, dataset_json, unpack_dataset, device)
        self.max_epochs = 5


class EffSparkMAETrainer_BS7(EffSparkMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, dataset_json, unpack_dataset, device)
        self.batch_size = 7
