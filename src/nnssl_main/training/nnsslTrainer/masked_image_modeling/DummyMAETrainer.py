from torch._C import device

from nnssl.experiment_planning.experiment_planners.plan import Plan

from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import BaseMAETrainer
from torch import device


class DummyMAETrainer(BaseMAETrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: device = ...,
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.num_epochs = 2  # Just do two epochs to test if writing also works as intended.
