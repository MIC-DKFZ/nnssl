from typing import Any

from torch import device
from torch import nn
from torch._C import device
from nnssl.architectures.spark_model import SparK3D
from nnssl.architectures.spark_utils import convert_to_einops_spark_cnn
from nnssl.experiment_planning.experiment_planners.plan import Plan
from nnssl.training.nnsslTrainer.masked_image_modeling.SparkTrainer import SparkMAETrainer
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet


class EinOps_SparkMAETrainer(SparkMAETrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: device = ...,
    ):
        super().__init__(plan, configuration_name, fold, dataset_json, unpack_dataset, device)

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

        spark_architecture = convert_to_einops_spark_cnn(network.encoder)
        network.encoder = spark_architecture
        actual_network = SparK3D(network, (160, 160, 160), self.use_mask_token)

        return actual_network


class EinOps_SparkMAETrainer_5ep_BS6(EinOps_SparkMAETrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: device = ...,
    ):
        plan.configurations[configuration_name].batch_size = 6
        super().__init__(plan, configuration_name, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 5
