from torch import nn


def _create_recon_proj_head(num_input_channels: int, features_per_stage: list[int] | tuple[int],
                            strides: list[list[int]]) -> nn.Sequential:
    """
    The official implementation of the decoder's architecture is hardcoded to fit the architecture of SwinUNETR,
    specifically the features per each stage.
    This reimplementation tries to be more flexible by mirroring the encoder stages.
    """
    recon_proj_head = nn.Sequential()
    features_per_upsample_stage = features_per_stage[::-1] + [num_input_channels]

    for i in range(len(features_per_upsample_stage) - 1):
        num_in_features, num_out_features = features_per_upsample_stage[i], features_per_upsample_stage[i+1]
        recon_proj_head.append(
            nn.Conv3d(num_in_features, num_out_features, kernel_size=3, stride=1, padding=1)
        )
        recon_proj_head.append(nn.InstanceNorm3d(num_out_features))
        recon_proj_head.append(nn.LeakyReLU())
        if strides[i][0] > 1:
            recon_proj_head.append(nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False))
    return recon_proj_head


class SwinUNETRArchitecture(nn.Module):
    def __init__(self, encoder: nn.Module, num_input_channels: int):
        super().__init__()
        self.encoder = encoder
        features_per_stage = encoder.output_channels
        strides = encoder.strides
        num_output_channels = features_per_stage[-1]

        self.rotation_proj_head = nn.Linear(num_output_channels, 4)
        self.contrast_proj_head = nn.Linear(num_output_channels, 512)

        # The phrasing in Section 4.1 of the paper (https://arxiv.org/abs/2111.14791) suggests the use of a
        # single transpose convolution layer to rescale to the initial input spatial resolution as implemented here:
        # https://github.com/Project-MONAI/research-contributions/blob/207cad9b2f15c958fcb5d9594ddaeca61f8f3dd6/SwinUNETR/Pretrain/models/ssl_head.py#L44
        # However the main training script uses this full decoder as the reconstruction head:
        # https://github.com/Project-MONAI/research-contributions/blob/207cad9b2f15c958fcb5d9594ddaeca61f8f3dd6/SwinUNETR/Pretrain/models/ssl_head.py#L54
        # Let's stick to the official repo ^^
        self.recon_proj_head = _create_recon_proj_head(num_input_channels, features_per_stage, strides)

    def forward(self, imgs):
        imgs_out = self.encoder(imgs)[-1]
        imgs_out_reshaped = imgs_out.flatten(start_dim=2, end_dim=4).transpose(1, 2)

        # for the rotation and contrast projection head, only one slice/channel is used
        # https://github.com/Project-MONAI/research-contributions/issues/87
        rotations_pred = self.rotation_proj_head(imgs_out_reshaped[:, 0])
        contrast_pred = self.contrast_proj_head(imgs_out_reshaped[:, 1])
        reconstructions = self.recon_proj_head(imgs_out)

        return rotations_pred, contrast_pred, reconstructions


