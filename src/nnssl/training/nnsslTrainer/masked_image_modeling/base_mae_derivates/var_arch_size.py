from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

from torch import nn

from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import (
    BaseMAETrainer_BS8_1000ep,
    BaseMAETrainer_BS1,
)


class BaseMAETrainer_BS8_ep1000_Arch_Width_B_Depth_B(BaseMAETrainer_BS8_1000ep):

    def build_architecture(self, *args, **kwargs) -> nn.Module:
        # Move to same plan as SPARK
        n_stages = 6
        network = ResidualEncoderUNet(
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
        return network


class BaseMAETrainer_BS8_ep1000_Arch_Width_M_Depth_B(BaseMAETrainer_BS8_1000ep):

    def build_architecture(self, *args, **kwargs) -> nn.Module:
        # Move to same plan as SPARK
        n_stages = 6
        network = ResidualEncoderUNet(
            input_channels=1,
            n_stages=n_stages,
            features_per_stage=[48, 96, 192, 384, 480, 480],  #
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
        return network


class BaseMAETrainer_BS8_ep1000_Arch_Width_L_Depth_B(BaseMAETrainer_BS8_1000ep):

    def build_architecture(self, *args, **kwargs) -> nn.Module:
        # Move to same plan as SPARK
        n_stages = 6
        network = ResidualEncoderUNet(
            input_channels=1,
            n_stages=n_stages,
            features_per_stage=[64, 128, 256, 512, 640, 640],  #
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
        return network


class BaseMAETrainer_BS8_ep1000_Arch_Width_B_Depth_M(BaseMAETrainer_BS8_1000ep):

    def build_architecture(self, *args, **kwargs) -> nn.Module:
        # Move to same plan as SPARK
        n_stages = 6
        network = ResidualEncoderUNet(
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
        return network


class BaseMAETrainer_BS8_ep1000_Arch_Width_B_Depth_L(BaseMAETrainer_BS8_1000ep):

    def build_architecture(self, *args, **kwargs) -> nn.Module:
        # Move to same plan as SPARK
        n_stages = 6
        network = ResidualEncoderUNet(
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
        return network


class BaseMAETrainer_BS8_ep1000_Arch_Width_M_Depth_M(BaseMAETrainer_BS8_1000ep):

    def build_architecture(self, *args, **kwargs) -> nn.Module:
        # Move to same plan as SPARK
        n_stages = 6
        network = ResidualEncoderUNet(
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
        return network


class BaseMAETrainer_BS8_ep1000_Arch_Width_B_Depth_B_LayerNorm(BaseMAETrainer_BS8_1000ep):

    def build_architecture(self, *args, **kwargs) -> nn.Module:
        # Move to same plan as SPARK
        n_stages = 6
        network = ResidualEncoderUNet(
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
        return network


class BaseMAETrainer_BS1_ep1000_Arch_Width_B_Depth_B_LayerNorm(BaseMAETrainer_BS1):

    def build_architecture(self, *args, **kwargs) -> nn.Module:
        # Move to same plan as SPARK
        n_stages = 6
        network = ResidualEncoderUNet(
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
        return network


class BaseMAETrainer_BS8_ep1000_Arch_Width_B_Depth_B_BatchNorm(BaseMAETrainer_BS8_1000ep):

    def build_architecture(self, *args, **kwargs) -> nn.Module:
        # Move to same plan as SPARK
        n_stages = 6
        network = ResidualEncoderUNet(
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
        return network


class BaseMAETrainer_BS1_ep1000_Arch_Width_B_Depth_B_BatchNorm(BaseMAETrainer_BS1):

    def build_architecture(self, *args, **kwargs) -> nn.Module:
        # Move to same plan as SPARK
        n_stages = 6
        network = ResidualEncoderUNet(
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
        return network
