from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.residual import BottleneckD
from torch import nn

from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans, DynamicArchitecturePlans
from nnssl.architectures.get_network_from_plan import get_network_from_plans
from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import (
    BaseMAETrainer_BS8_1000ep,
    BaseMAETrainer_BS1,
)


class BaseMAETrainer_BS8_ep1000_Arch_Width_B_Depth_B(BaseMAETrainer_BS8_1000ep):

    def build_architecture_and_adaptation_plan(self, *args, **kwargs) -> nn.Module:
        # Move to same plan as SPARK
        n_stages = 6
        arch_kwargs = DynamicArchitecturePlans(
            n_stages=n_stages,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
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
        )

        network_old = ResidualEncoderUNet(
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

        assert network.state_dict().keys() == network_old.state_dict().keys(), "State dicts do not match"
        for k in network.state_dict().keys():
            if network.state_dict()[k].shape != network_old.state_dict()[k].shape:
                print(
                    f"Key {k} has different shape: {network.state_dict()[k].shape} vs {network_old.state_dict()[k].shape}"
                )
            else:
                pass
        return network, adapt_plan


class BaseMAETrainer_BS8_ep1000_Arch_Width_M_Depth_B(BaseMAETrainer_BS8_1000ep):

    def build_architecture_and_adaptation_plan(self, *args, **kwargs) -> nn.Module:
        # Move to same plan as SPARK
        n_stages = 6
        arch_kwargs = DynamicArchitecturePlans(
            n_stages=n_stages,
            features_per_stage=[48, 96, 192, 384, 480, 480],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
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
        )

        network_old = ResidualEncoderUNet(
            input_channels=1,
            n_stages=n_stages,
            features_per_stage=[48, 96, 192, 384, 480, 480],
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

        assert network.state_dict().keys() == network_old.state_dict().keys(), "State dicts do not match"
        for k in network.state_dict().keys():
            if network.state_dict()[k].shape != network_old.state_dict()[k].shape:
                print(
                    f"Key {k} has different shape: {network.state_dict()[k].shape} vs {network_old.state_dict()[k].shape}"
                )
            else:
                pass
        return network, adapt_plan


class BaseMAETrainer_BS8_ep1000_Arch_Width_L_Depth_B(BaseMAETrainer_BS8_1000ep):

    def build_architecture_and_adaptation_plan(self, *args, **kwargs) -> nn.Module:
        # Move to same plan as SPARK
        n_stages = 6
        arch_kwargs = DynamicArchitecturePlans(
            n_stages=n_stages,
            features_per_stage=[64, 128, 256, 512, 640, 640],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
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
        )

        network_old = ResidualEncoderUNet(
            input_channels=1,
            n_stages=n_stages,
            features_per_stage=[64, 128, 256, 512, 640, 640],
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

        assert network.state_dict().keys() == network_old.state_dict().keys(), "State dicts do not match"
        for k in network.state_dict().keys():
            if network.state_dict()[k].shape != network_old.state_dict()[k].shape:
                print(
                    f"Key {k} has different shape: {network.state_dict()[k].shape} vs {network_old.state_dict()[k].shape}"
                )
            else:
                pass
        return network, adapt_plan


class BaseMAETrainer_BS8_ep1000_Arch_Width_B_Depth_M(BaseMAETrainer_BS8_1000ep):

    def build_architecture_and_adaptation_plan(self, *args, **kwargs) -> nn.Module:
        # Move to same plan as SPARK
        n_stages = 6
        arch_kwargs = DynamicArchitecturePlans(
            n_stages=n_stages,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 4, 6, 8, 8, 8],
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
        )

        network_old = ResidualEncoderUNet(
            input_channels=1,
            n_stages=n_stages,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 4, 6, 8, 8, 8],
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
        return network, adapt_plan


class BaseMAETrainer_BS8_ep1000_Arch_Width_B_Depth_L(BaseMAETrainer_BS8_1000ep):
    def build_architecture_and_adaptation_plan(self, *args, **kwargs) -> nn.Module:
        # Move to same plan as SPARK
        n_stages = 6
        arch_kwargs = DynamicArchitecturePlans(
            n_stages=n_stages,
            features_per_stage=[32, 64, 128, 256, 320, 320],
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
        )

        network_old = ResidualEncoderUNet(
            input_channels=1,
            n_stages=n_stages,
            features_per_stage=[32, 64, 128, 256, 320, 320],
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
        return network, adapt_plan


class BaseMAETrainer_BS8_ep1000_Arch_Width_M_Depth_M(BaseMAETrainer_BS8_1000ep):

    def build_architecture_and_adaptation_plan(self, *args, **kwargs) -> nn.Module:
        # Move to same plan as SPARK
        n_stages = 6
        arch_kwargs = DynamicArchitecturePlans(
            n_stages=n_stages,
            features_per_stage=[48, 96, 192, 384, 480, 480],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 4, 6, 8, 8, 8],
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
        )

        network_old = ResidualEncoderUNet(
            input_channels=1,
            n_stages=n_stages,
            features_per_stage=[48, 96, 192, 384, 480, 480],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 4, 6, 8, 8, 8],
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
        return network, adapt_plan


class BaseMAETrainer_BS8_ep1000_Arch_Width_B_Depth_B_LayerNorm(BaseMAETrainer_BS8_1000ep):

    def build_architecture_and_adaptation_plan(self, *args, **kwargs) -> nn.Module:
        # Move to same plan as SPARK
        n_stages = 6
        arch_kwargs = DynamicArchitecturePlans(
            n_stages=n_stages,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 4, 6, 8, 8, 8],
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=nn.LayerNorm,
            norm_op_kwargs={"eps": 1e-5},
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
        )

        network_old = ResidualEncoderUNet(
            input_channels=1,
            n_stages=n_stages,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 4, 6, 8, 8, 8],
            num_classes=1,
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=nn.LayerNorm,
            norm_op_kwargs={"eps": 1e-5},
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
        return network, adapt_plan


class BaseMAETrainer_BS1_ep1000_Arch_Width_B_Depth_B_LayerNorm(BaseMAETrainer_BS1):

    def build_architecture_and_adaptation_plan(self, *args, **kwargs) -> nn.Module:
        # Move to same plan as SPARK
        n_stages = 6
        arch_kwargs = DynamicArchitecturePlans(
            n_stages=n_stages,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 4, 6, 8, 8, 8],
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=nn.LayerNorm,
            norm_op_kwargs={"eps": 1e-5},
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
        )

        network_old = ResidualEncoderUNet(
            input_channels=1,
            n_stages=n_stages,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 4, 6, 8, 8, 8],
            num_classes=1,
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=nn.LayerNorm,
            norm_op_kwargs={"eps": 1e-5},
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
        return network, adapt_plan


class BaseMAETrainer_BS8_ep1000_Arch_Width_B_Depth_B_BatchNorm(BaseMAETrainer_BS8_1000ep):

    def build_architecture_and_adaptation_plan(self, *args, **kwargs) -> nn.Module:
        # Move to same plan as SPARK
        n_stages = 6
        arch_kwargs = DynamicArchitecturePlans(
            n_stages=n_stages,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 4, 6, 8, 8, 8],
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=nn.BatchNorm3d,
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
        )

        network_old = ResidualEncoderUNet(
            input_channels=1,
            n_stages=n_stages,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 4, 6, 8, 8, 8],
            num_classes=1,
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=nn.BatchNorm3d,
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
        return network, adapt_plan


class BaseMAETrainer_BS1_ep1000_Arch_Width_B_Depth_B_BatchNorm(BaseMAETrainer_BS1):

    def build_architecture_and_adaptation_plan(self, *args, **kwargs) -> nn.Module:
        # Move to same plan as SPARK
        n_stages = 6
        arch_kwargs = DynamicArchitecturePlans(
            n_stages=n_stages,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 4, 6, 8, 8, 8],
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=nn.BatchNorm3d,
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
        )

        network_old = ResidualEncoderUNet(
            input_channels=1,
            n_stages=n_stages,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 4, 6, 8, 8, 8],
            num_classes=1,
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=nn.BatchNorm3d,
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
        return network, adapt_plan


class BaseMAETrainer_BS8_ep1000_Arch_Width_B_Depth_B_Drop005(BaseMAETrainer_BS8_ep1000_Arch_Width_B_Depth_B):

    def build_architecture_and_adaptation_plan(self, *args, **kwargs) -> nn.Module:
        # Move to same plan as SPARK
        n_stages = 6
        arch_kwargs = DynamicArchitecturePlans(
            n_stages=n_stages,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 4, 6, 8, 8, 8],
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5},
            dropout_op=nn.Dropout3d,
            dropout_op_kwargs={"p": 0.05},
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
            allow_init=False,
        )
        arch_plan = ArchitecturePlans(arch_class_name="ResidualEncoderUNet", arch_kwargs=arch_kwargs)
        adapt_plan = AdaptationPlan(
            architecture_plans=arch_plan,
            pretrain_plan=self.plan,
            recommended_downstream_patchsize=self.recommended_downstream_patchsize,
            pretrain_num_input_channels=1,
            key_to_encoder="encoder.stages",
            key_to_stem="encoder.stem",
        )

        network_old = ResidualEncoderUNet(
            input_channels=1,
            n_stages=n_stages,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 4, 6, 8, 8, 8],
            num_classes=1,
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5},
            dropout_op=nn.Dropout3d,
            dropout_op_kwargs={"p": 0.05},
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
        return network, adapt_plan


class BaseMAETrainer_BS8_ep1000_Arch_Width_B_Depth_B_Drop01(BaseMAETrainer_BS8_ep1000_Arch_Width_B_Depth_B):

    def build_architecture_and_adaptation_plan(self, *args, **kwargs) -> nn.Module:
        # Move to same plan as SPARK
        n_stages = 6
        arch_kwargs = DynamicArchitecturePlans(
            n_stages=n_stages,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 4, 6, 8, 8, 8],
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5},
            dropout_op=nn.Dropout3d,
            dropout_op_kwargs={"p": 0.1},
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
            allow_init=False,
        )
        arch_plan = ArchitecturePlans(arch_class_name="ResidualEncoderUNet", arch_kwargs=arch_kwargs)
        adapt_plan = AdaptationPlan(
            architecture_plans=arch_plan,
            pretrain_plan=self.plan,
            recommended_downstream_patchsize=self.recommended_downstream_patchsize,
            pretrain_num_input_channels=1,
            key_to_encoder="encoder.stages",
            key_to_stem="encoder.stem",
        )

        network_old = ResidualEncoderUNet(
            input_channels=1,
            n_stages=n_stages,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 4, 6, 8, 8, 8],
            num_classes=1,
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5},
            dropout_op=nn.Dropout3d,
            dropout_op_kwargs={"p": 0.1},
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
        return network, adapt_plan


class BaseMAETrainer_BS8_ep1000_Arch_Width_B_Depth_B_Drop015(BaseMAETrainer_BS8_ep1000_Arch_Width_B_Depth_B):

    def build_architecture_and_adaptation_plan(self, *args, **kwargs) -> nn.Module:
        # Move to same plan as SPARK
        n_stages = 6
        arch_kwargs = DynamicArchitecturePlans(
            n_stages=n_stages,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 4, 6, 8, 8, 8],
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5},
            dropout_op=nn.Dropout3d,
            dropout_op_kwargs={"p": 0.15},
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
            allow_init=False,
        )
        arch_plan = ArchitecturePlans(arch_class_name="ResidualEncoderUNet", arch_kwargs=arch_kwargs)
        adapt_plan = AdaptationPlan(
            architecture_plans=arch_plan,
            pretrain_plan=self.plan,
            pretrain_num_input_channels=1,
            recommended_downstream_patchsize=self.recommended_downstream_patchsize,
            key_to_encoder="encoder.stages",
            key_to_stem="encoder.stem",
        )

        network_old = ResidualEncoderUNet(
            input_channels=1,
            n_stages=n_stages,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 4, 6, 8, 8, 8],
            num_classes=1,
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5},
            dropout_op=nn.Dropout3d,
            dropout_op_kwargs={"p": 0.15},
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
        return network, adapt_plan


class BaseMAETrainer_BS8_ep1000_Arch_Width_B_Depth_B_Drop02(BaseMAETrainer_BS8_ep1000_Arch_Width_B_Depth_B):

    def build_architecture_and_adaptation_plan(self, *args, **kwargs) -> nn.Module:
        # Move to same plan as SPARK
        n_stages = 6
        arch_kwargs = DynamicArchitecturePlans(
            n_stages=n_stages,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 4, 6, 8, 8, 8],
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5},
            dropout_op=nn.Dropout3d,
            dropout_op_kwargs={"p": 0.2},
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
            allow_init=False,
        )
        arch_plan = ArchitecturePlans(arch_class_name="ResidualEncoderUNet", arch_kwargs=arch_kwargs)
        adapt_plan = AdaptationPlan(
            architecture_plans=arch_plan,
            pretrain_plan=self.plan,
            recommended_downstream_patchsize=self.recommended_downstream_patchsize,
            pretrain_num_input_channels=1,
            key_to_encoder="encoder.stages",
            key_to_stem="encoder.stem",
        )

        network_old = ResidualEncoderUNet(
            input_channels=1,
            n_stages=n_stages,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 4, 6, 8, 8, 8],
            num_classes=1,
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5},
            dropout_op=nn.Dropout3d,
            dropout_op_kwargs={"p": 0.2},
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
        return network, adapt_plan


class BaseMAETrainer_BS8_ep1000_Arch_Width_B_Depth_B_Bneck(BaseMAETrainer_BS8_ep1000_Arch_Width_B_Depth_B):

    def build_architecture_and_adaptation_plan(self, *args, **kwargs) -> nn.Module:
        # Move to same plan as SPARK
        n_stages = 6
        arch_kwargs = DynamicArchitecturePlans(
            n_stages=n_stages,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 4, 6, 8, 8, 8],
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            block=BottleneckD,
            bottleneck_channels=[64, 128, 256, 512, 640, 640],
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
        )

        network_old = ResidualEncoderUNet(
            input_channels=1,
            n_stages=n_stages,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 4, 6, 8, 8, 8],
            num_classes=1,
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=False,
            block=BottleneckD,
            bottleneck_channels=[64, 128, 256, 512, 640, 640],
        )

        assert network.state_dict().keys() == network_old.state_dict().keys(), "State dicts do not match"
        for k in network.state_dict().keys():
            if network.state_dict()[k].shape != network_old.state_dict()[k].shape:
                print(
                    f"Key {k} has different shape: {network.state_dict()[k].shape} vs {network_old.state_dict()[k].shape}"
                )
            else:
                pass
        return network, adapt_plan
