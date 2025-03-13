from typing import Any

from torch import device
from torch import nn
from torch._C import device
from nnssl.architectures.get_network_by_name import get_network_by_name
from nnssl.architectures.spark_model import SparK3D
from nnssl.architectures.spark_utils import convert_to_einops_spark_cnn
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.training.nnsslTrainer.masked_image_modeling.SparkTrainer import SparkMAETrainer
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet


class EinOps_SparkMAETrainer(SparkMAETrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: device = ...,
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)

    def build_architecture(
        self, config_plan: ConfigurationPlan, num_input_channels: int, num_output_channels: int
    ) -> nn.Module:
        n_stages = 6
        network = get_network_by_name(
            config_plan,
            "ResEncL",
            num_input_channels,
            num_output_channels,
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
        pretrain_json: dict,
        device: device = ...,
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 6
        self.num_epochs = 5
