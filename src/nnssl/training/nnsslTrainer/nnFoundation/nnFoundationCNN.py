import torch
from torch import nn

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans, DynamicArchitecturePlans
from nnssl.architectures.get_network_from_plan import get_network_from_plans
from nnssl.experiment_planning.experiment_planners.plan import Plan
from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import BaseMAETrainer_BS8_1000ep


class BaseMAETrainer_BS8_ep2500_ps192_bs8_lr_1e2(BaseMAETrainer_BS8_1000ep):
    def __init__(
            self,
            plan: Plan,
            configuration_name: str,
            fold: int,
            pretrain_json: dict,
            device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.num_epochs = 2500
        self.config_plan.patch_size = (192, 192, 192)
        self.recommended_downstream_patchsize = (192, 192, 192)
        self.initial_lr = 1e-2


class BaseMAETrainer_BS8_ep2500_ps192_Arch_Width_L_Depth_L_bs96_lr_1e2(BaseMAETrainer_BS8_ep2500_ps192_bs8_lr_1e2):
    def __init__(
            self,
            plan: Plan,
            configuration_name: str,
            fold: int,
            pretrain_json: dict,
            device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 96

    def build_architecture_and_adaptation_plan(self, *args, **kwargs) -> nn.Module:
        # Move to same plan as SPARK
        n_stages = 6
        arch_kwargs = DynamicArchitecturePlans(
            n_stages=n_stages,
            features_per_stage=[64, 128, 256, 512, 640, 640],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[2, 5, 6, 10, 10, 12],
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
        )
        network = get_network_from_plans(
            "ResidualEncoderUNet",
            arch_kwargs.serialize(),
            arch_kwargs_req_import=arch_kwargs.get_kwargs_requiring_import(),
            input_channels=1,
            output_channels=1,
            deep_supervision=False,
        )
        arch_plan = ArchitecturePlans(arch_class_name="ResidualEncoderUNet", arch_kwargs=arch_kwargs)
        adapt_plan = AdaptationPlan(
            architecture_plans=arch_plan,
            pretrain_plan=self.plan,
            recommended_downstream_patchsize=self.recommended_downstream_patchsize,
            pretrain_num_input_channels=1,
            key_to_encoder="encoder.stages",
            key_to_stem="encoder.stem",
            keys_to_in_proj=("encoder.stem.convs.0.conv", "encoder.stem.convs.0.all_modules.0"),
        )

        network_old = ResidualEncoderUNet(
            input_channels=1,
            n_stages=n_stages,
            features_per_stage=[64, 128, 256, 512, 640, 640],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[2, 5, 6, 10, 10, 12],
            num_classes=1,
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=False,
        )

        assert network.state_dict().keys() == network_old.state_dict().keys(), "State dicts do not match"
        for k in network.state_dict().keys():
            if network.state_dict()[k].shape != network_old.state_dict()[k].shape:
                print(
                    f"Key {k} has different shape: {network.state_dict()[k].shape} vs {network_old.state_dict()[k].shape}"
                )
            else:
                pass
        n_params = sum(p.numel() for p in network.parameters())
        self.print_to_log_file(f"Number of parameters: {n_params}")
        return network, adapt_plan


class nnFoundationCNN_trainer(BaseMAETrainer_BS8_ep2500_ps192_Arch_Width_L_Depth_L_bs96_lr_1e2):
    """Alias for the nnFoundationCNN pre-training."""
