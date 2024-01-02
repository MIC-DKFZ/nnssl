from torch._C import device
from nnssl.architectures.build_architecture import build_network_architecture
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.training.loss.mse_loss import MSELoss
from nnssl.training.nnsslTrainer.nnsslAbstractMAETrainer import nnAbstractMAETrainer
from torch import device, nn


class nnsslDummyMAETrainer(nnAbstractMAETrainer):
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
        self.num_epochs = 2  # Just do two epochs to test if writing also works as intended.

    def _build_loss(self):
        """
        This is where you build your loss function. You can use anything from torch.nn here
        :return:
        """
        # return nn.MSELoss()
        return MSELoss()

    def build_architecture(
        self, config_plan: ConfigurationPlan, num_input_channels: int, num_output_channels: int
    ) -> nn.Module:
        architecture = build_network_architecture(config_plan, num_input_channels, num_output_channels)
        return architecture
