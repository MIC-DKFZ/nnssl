from nnssl.architectures.build_architecture import build_network_architecture
from nnssl.architectures.convert_to_spark import convert_to_spark_cnn
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan
from nnssl.training.nnsslTrainer.AbstractMAETrainer import AbstractMAETrainer
from nnssl.training.loss.spark_loss import SparkLoss
from torch import nn


class SparkMAETrainer(AbstractMAETrainer):
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
