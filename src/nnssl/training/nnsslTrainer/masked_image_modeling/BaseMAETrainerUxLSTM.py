"""
BaseMAETrainerUxLSTM: MAE SSL pretraining with UxLSTMBot backbone.

UxLSTMBot architecture:
  ResEncoder (CNN) -> ViLLayer (xLSTM bottleneck) -> ResDecoder

Usage:
    nnssl_train BaseMAETrainerUxLSTM Dataset910_Combined onemmiso

Author: Abdul Qayyum
"""

import sys
import torch
import torch.nn as nn
from typing import Tuple
from typing_extensions import override

# Add Scar_Segmentation_models to path for UxLSTMBot
sys.path.insert(0, '/home/aqayyum/Scar_Segmentation_models')

from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import BaseMAETrainer
from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan
from batchgenerators.utilities.file_and_folder_operations import save_json

from nnunetv2.nets.UxLSTMBot_3d import UXlstmBot
from nnunetv2.utilities.network_initialization import InitWeights_He


def get_uxlstm_bot(
    num_input_channels: int,
    num_output_channels: int,
    deep_supervision: bool = False,
) -> UXlstmBot:
    """
    UxLSTMBot with same feature dims as ResEncL:
      6 stages, [32, 64, 128, 256, 320, 320]
      ViLLayer (xLSTM) at bottleneck (320ch)
    """
    n_stages = 6
    network = UXlstmBot(
        input_channels=num_input_channels,
        n_stages=n_stages,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        n_conv_per_stage=[1, 3, 4, 6, 6, 6],
        num_classes=num_output_channels,
        n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=deep_supervision,
    )
    network.apply(InitWeights_He(1e-2))
    return network


class BaseMAETrainerUxLSTM(BaseMAETrainer):
    """
    MAE SSL pretraining with UxLSTMBot backbone.

    vs BaseMAETrainer:
      Backbone:    ResEncL (pure CNN) -> UXlstmBot (CNN + xLSTM)
      Bottleneck:  conv blocks only  -> ViLLayer (Vision-LSTM)
    """

    @override
    def build_architecture_and_adaptation_plan(
        self,
        config_plan: ConfigurationPlan,
        num_input_channels: int,
        num_output_channels: int,
        *args,
        **kwargs,
    ) -> Tuple[nn.Module, AdaptationPlan]:

        architecture = get_uxlstm_bot(
            num_input_channels,
            num_output_channels,
            deep_supervision=False,
        )

        arch_plans = ArchitecturePlans(arch_class_name="UXlstmBot")
        adapt_plan = AdaptationPlan(
            architecture_plans=arch_plans,
            pretrain_plan=self.plan,
            pretrain_num_input_channels=num_input_channels,
            recommended_downstream_patchsize=self.recommended_downstream_patchsize,
            key_to_encoder="encoder.stages",
            key_to_stem="encoder.stem",
            keys_to_in_proj=(
                "encoder.stem.convs.0.conv",
                "encoder.stem.convs.0.all_modules.0",
            ),
            key_to_xlstm="xlstm",  # Transfer xLSTM bottleneck weights too
        )
        save_json(adapt_plan.serialize(), self.adaptation_json_plan)
        return architecture, adapt_plan


class BaseMAETrainerUxLSTM_BS2(BaseMAETrainerUxLSTM):
    def __init__(self, plan, configuration_name, fold, pretrain_json,
                 device=torch.device("cuda")):
        plan.configurations[configuration_name].batch_size = 2
        super().__init__(plan, configuration_name, fold, pretrain_json, device)


class BaseMAETrainerUxLSTM_BS2_ep1000(BaseMAETrainerUxLSTM):
    def __init__(self, plan, configuration_name, fold, pretrain_json,
                 device=torch.device("cuda")):
        plan.configurations[configuration_name].batch_size = 2
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.num_epochs = 1000


class BaseMAETrainerUxLSTM_BS4_ep1000(BaseMAETrainerUxLSTM):
    def __init__(self, plan, configuration_name, fold, pretrain_json,
                 device=torch.device("cuda")):
        plan.configurations[configuration_name].batch_size = 4
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.num_epochs = 1000


class BaseMAETrainerUxLSTM_BS1_patch128(BaseMAETrainerUxLSTM):
    """UxLSTM MAE, BS=1, patch 128^3 to fit in GPU memory."""
    def __init__(self, plan, configuration_name, fold, pretrain_json,
                 device=torch.device("cuda")):
        plan.configurations[configuration_name].batch_size = 1
        plan.configurations[configuration_name].patch_size = (128, 128, 128)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
