import torch
from torch import nn
from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans
from nnssl.architectures.get_network_by_name import get_network_by_name
from nnssl.architectures.spark_model import EfficientSpark3D
from nnssl.architectures.spark_utils import convert_to_spark_cnn
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.training.nnsslTrainer.masked_image_modeling.SparkTrainer import SparkMAETrainer
from batchgenerators.utilities.file_and_folder_operations import save_json


class EffSparkMAETrainer(SparkMAETrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.network: EfficientSpark3D = ...

    def build_architecture_and_adaptation_plan(
        self,
        config_plan: ConfigurationPlan,
        num_input_channels: int,
        num_output_channels: int,
    ) -> nn.Module:
        # ----------------------------- Network creation ----------------------------- #
        network = get_network_by_name(
            config_plan,
            "ResEncL",
            num_input_channels,
            num_output_channels,
            deep_supervision=False,
        )

        spark_architecture = convert_to_spark_cnn(network.encoder)
        network.encoder = spark_architecture
        actual_network = EfficientSpark3D(network)
        # ------------------------------ Adaptation Plan ----------------------------- #

        adapt_plan = AdaptationPlan(
            architecture_plans=ArchitecturePlans("ResEncL"),
            pretrain_plan=self.plan,
            pretrain_num_input_channels=1,
            recommended_downstream_patchsize=self.recommended_downstream_patchsize,
            key_to_encoder="encoder.stages",
            key_to_stem="encoder.stem",
            keys_to_in_proj=("encoder.stem.convs.0.conv", "encoder.stem.convs.0.all_modules.0"),
        )

        return actual_network, adapt_plan


class EffSparkMAETrainer_BS8_1000ep(EffSparkMAETrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 8
        self.num_epochs = 1000


class EffSparkMAETrainer_BS6_1000ep(EffSparkMAETrainer):

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


class EffSparkMAETrainer_5ep(EffSparkMAETrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.num_epochs = 5


class EffSparkMAETrainer_5ep_BS6(EffSparkMAETrainer_5ep):

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


class EffSparkMAETrainer_BS8_5ep(EffSparkMAETrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 8
        self.num_epochs = 5


class EffSparkMAETrainer_BS7(EffSparkMAETrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 7


class EffSparkMAETrainer_BS7_LR_5e2(EffSparkMAETrainer_BS7):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.initial_lr = 5e-2


class EffSparkMAETrainer_BS7_LR_3e2(EffSparkMAETrainer_BS7):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.initial_lr = 3e-2


class EffSparkMAETrainer_BS28_LR_3e2(EffSparkMAETrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 28
        self.initial_lr = 3e-2


class EffSparkMAETrainer_BS6_LR_5e2(EffSparkMAETrainer):

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
        self.initial_lr = 5e-2


class EffSparkMAETrainer_BS7_LR_3e2_Mask40(EffSparkMAETrainer_BS7_LR_3e2):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.mask_percentage = 0.4


class EffSparkMAETrainer_BS7_LR_3e2_Mask60(EffSparkMAETrainer_BS7_LR_3e2):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.mask_percentage = 0.6


class EffSparkMAETrainer_BS6_LR_3e2_Mask30(EffSparkMAETrainer):

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
        self.initial_lr = 3e-2
        self.mask_percentage = 0.30


class EffSparkMAETrainer_BS6_LR_3e2_Mask45(EffSparkMAETrainer_BS6_LR_3e2_Mask30):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.mask_percentage = 0.45


class EffSparkMAETrainer_BS6_LR_3e2_Mask60(EffSparkMAETrainer_BS6_LR_3e2_Mask30):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.mask_percentage = 0.6


class EffSparkMAETrainer_BS6_LR_3e2_Mask75(EffSparkMAETrainer_BS6_LR_3e2_Mask30):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.mask_percentage = 0.75


class EffSparkMAETrainer_BS6_LR_3e2_Mask90(EffSparkMAETrainer_BS6_LR_3e2_Mask30):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.mask_percentage = 0.9
