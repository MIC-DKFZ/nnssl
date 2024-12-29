import torch
from nnssl.experiment_planning.experiment_planners.plan import Plan
from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import BaseMAETrainer_BS8_1000ep


class BaseMAETrainer_BS8_1000ep_LR1e2_WD_1e5(BaseMAETrainer_BS8_1000ep):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        # Defaults:
        # self.initial_lr = 1e-2
        # self.weight_decay = 3e-5
        self.initial_lr = 1e-2
        self.weight_decay = 1e-5


class BaseMAETrainer_BS8_1000ep_LR1e2_WD_7e5(BaseMAETrainer_BS8_1000ep):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        # Defaults:
        # self.initial_lr = 1e-2
        # self.weight_decay = 3e-5
        self.initial_lr = 1e-2
        self.weight_decay = 7e-5


class BaseMAETrainer_BS8_1000ep_LR1e2_WD_1e4(BaseMAETrainer_BS8_1000ep):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        # Defaults:
        # self.initial_lr = 1e-2
        # self.weight_decay = 3e-5
        self.initial_lr = 1e-2
        self.weight_decay = 1e-4


class BaseMAETrainer_BS8_1000ep_LR1e2_WD_3e4(BaseMAETrainer_BS8_1000ep):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        # Defaults:
        # self.initial_lr = 1e-2
        # self.weight_decay = 3e-5
        self.initial_lr = 1e-2
        self.weight_decay = 3e-4


class BaseMAETrainer_BS8_1000ep_LR1e2_WD_7e4(BaseMAETrainer_BS8_1000ep):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        # Defaults:
        # self.initial_lr = 1e-2
        # self.weight_decay = 3e-5
        self.initial_lr = 1e-2
        self.weight_decay = 7e-4


class BaseMAETrainer_BS8_1000ep_LR1e2_WD_1e3(BaseMAETrainer_BS8_1000ep):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        # Defaults:
        # self.initial_lr = 1e-2
        # self.weight_decay = 3e-5
        self.initial_lr = 1e-2
        self.weight_decay = 1e-3


class BaseMAETrainer_BS8_1000ep_LR3e2_WD_3e5(BaseMAETrainer_BS8_1000ep):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        # Defaults:
        # self.initial_lr = 1e-2
        # self.weight_decay = 3e-5
        self.initial_lr = 3e-2
        self.weight_decay = 3e-5


class BaseMAETrainer_BS8_1000ep_LR7e2_WD_3e5(BaseMAETrainer_BS8_1000ep):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        # Defaults:
        # self.initial_lr = 1e-2
        # self.weight_decay = 3e-5
        self.initial_lr = 7e-2
        self.weight_decay = 3e-5


class BaseMAETrainer_BS8_1000ep_LR7e3_WD_3e5(BaseMAETrainer_BS8_1000ep):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        # Defaults:
        # self.initial_lr = 1e-2
        # self.weight_decay = 3e-5
        self.initial_lr = 7e-3
        self.weight_decay = 3e-5


class BaseMAETrainer_BS8_1000ep_LR3e3_WD_3e5(BaseMAETrainer_BS8_1000ep):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        # Defaults:
        # self.initial_lr = 1e-2
        # self.weight_decay = 3e-5
        self.initial_lr = 3e-3
        self.weight_decay = 3e-5


class BaseMAETrainer_BS8_1000ep_LR1e3_WD_3e5(BaseMAETrainer_BS8_1000ep):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        # Defaults:
        # self.initial_lr = 1e-2
        # self.weight_decay = 3e-5
        self.initial_lr = 1e-3
        self.weight_decay = 3e-5


class BaseMAETrainer_BS8_1000ep_LR3e2_WD_1e3(BaseMAETrainer_BS8_1000ep):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        # Defaults:
        # self.initial_lr = 1e-2
        # self.weight_decay = 3e-5
        self.initial_lr = 3e-2
        self.weight_decay = 1e-3


class BaseMAETrainer_BS8_1000ep_LR7e2_WD_1e3(BaseMAETrainer_BS8_1000ep):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        # Defaults:
        # self.initial_lr = 1e-2
        # self.weight_decay = 3e-5
        self.initial_lr = 7e-2
        self.weight_decay = 1e-3


class BaseMAETrainer_BS8_1000ep_LR7e3_WD_1e3(BaseMAETrainer_BS8_1000ep):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        # Defaults:
        # self.initial_lr = 1e-2
        # self.weight_decay = 3e-5
        self.initial_lr = 7e-3
        self.weight_decay = 1e-3


class BaseMAETrainer_BS8_1000ep_LR3e3_WD_1e3(BaseMAETrainer_BS8_1000ep):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        # Defaults:
        # self.initial_lr = 1e-2
        # self.weight_decay = 3e-5
        self.initial_lr = 3e-3
        self.weight_decay = 1e-3


class BaseMAETrainer_BS8_1000ep_LR1e3_WD_1e3(BaseMAETrainer_BS8_1000ep):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        # Defaults:
        # self.initial_lr = 1e-2
        # self.weight_decay = 3e-5
        self.initial_lr = 1e-3
        self.weight_decay = 1e-3


class BaseMAETrainer_BS8_1000ep_LR3e2_WD_3e4(BaseMAETrainer_BS8_1000ep):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        # Defaults:
        # self.initial_lr = 1e-2
        # self.weight_decay = 3e-5
        self.initial_lr = 3e-2
        self.weight_decay = 3e-4


class BaseMAETrainer_BS8_1000ep_LR7e2_WD_3e4(BaseMAETrainer_BS8_1000ep):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        # Defaults:
        # self.initial_lr = 1e-2
        # self.weight_decay = 3e-5
        self.initial_lr = 7e-2
        self.weight_decay = 3e-4


class BaseMAETrainer_BS8_1000ep_LR7e3_WD_3e4(BaseMAETrainer_BS8_1000ep):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        # Defaults:
        # self.initial_lr = 1e-2
        # self.weight_decay = 3e-5
        self.initial_lr = 7e-3
        self.weight_decay = 3e-4


class BaseMAETrainer_BS8_1000ep_LR3e3_WD_3e4(BaseMAETrainer_BS8_1000ep):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        # Defaults:
        # self.initial_lr = 1e-2
        # self.weight_decay = 3e-5
        self.initial_lr = 3e-3
        self.weight_decay = 3e-4


class BaseMAETrainer_BS8_1000ep_LR1e3_WD_3e4(BaseMAETrainer_BS8_1000ep):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        # Defaults:
        # self.initial_lr = 1e-2
        # self.weight_decay = 3e-5
        self.initial_lr = 1e-3
        self.weight_decay = 3e-4
