import torch
from nnssl.architectures.get_network_by_name import get_network_by_name
from nnssl.architectures.spark_model import SparK3D
from nnssl.architectures.spark_utils import convert_to_spark_cnn

from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.training.loss.spark_loss import SparkLoss
from torch import nn
from nnssl.training.nnsslTrainer.masked_image_modeling.SparkTrainer import SparkMAETrainer
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet


class NoTokenSparkMAETrainer(SparkMAETrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.mask_percentage: float = 0.75
        self.loss: SparkLoss
        self.stop_at_nans = True
        self.use_mask_token: bool = True
        self.network: SparK3D = ...

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
        actual_network = SparK3D(network, use_mask_token=False)
        raise NotImplementedError("Missing adaptation plan")
        return actual_network


class NoTokenSparkMAETrainer_BS6_1000ep(NoTokenSparkMAETrainer):

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
        self.num_epochs = 1000


class NoTokenSparkMAETrainer_5ep_BS6(NoTokenSparkMAETrainer):

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
