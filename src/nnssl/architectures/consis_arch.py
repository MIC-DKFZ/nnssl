from typing import Union, List, Tuple, Type

import torch
import numpy as np
import torch.nn as nn

from einops import rearrange
from torch.nn.init import trunc_normal_
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.eva import Eva
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.simple_conv_blocks import (
    StackedConvBlocks,
)
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.residual_encoders import (
    ResidualEncoder,
)
from dynamic_network_architectures.building_blocks.plain_conv_encoder import (
    PlainConvEncoder,
)

from nnssl.architectures.evaMAE_module import EvaMAE


class UNetDecoder(nn.Module):
    def __init__(
        self,
        encoder: Union[PlainConvEncoder, ResidualEncoder],
        num_classes: int,
        n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
        deep_supervision,
        latents: bool = False,
        nonlin_first: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        conv_bias: bool = None,
    ):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.latents = latents

        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, (
            "n_conv_per_stage must have as many entries as we have "
            "resolution stages - 1 (n_stages in encoder - 1), "
            "here: %d" % n_stages_encoder
        )

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = (
            encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        )
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = (
            encoder.dropout_op_kwargs
            if dropout_op_kwargs is None
            else dropout_op_kwargs
        )
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = (
            encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs
        )

        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(
                transpconv_op(
                    input_features_below,
                    input_features_skip,
                    stride_for_transpconv,
                    stride_for_transpconv,
                    bias=conv_bias,
                )
            )
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(
                StackedConvBlocks(
                    n_conv_per_stage[s - 1],
                    encoder.conv_op,
                    2 * input_features_skip,
                    input_features_skip,
                    encoder.kernel_sizes[-(s + 1)],
                    1,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    nonlin_first,
                )
            )

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(
                encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True)
            )

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                if self.latents:
                    seg_outputs.append(x)
                else:
                    seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs, self.seg_layers[-1](seg_outputs[0])
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append(
                [i // j for i, j in zip(input_size, self.encoder.strides[s])]
            )
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s + 1)])
            # trans conv
            output += np.prod(
                [self.encoder.output_channels[-(s + 2)], *skip_sizes[-(s + 1)]],
                dtype=np.int64,
            )
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod(
                    [self.num_classes, *skip_sizes[-(s + 1)]], dtype=np.int64
                )
        return output


class ConsisMAE(ResidualEncoderUNet):

    def __init__(
        self,
        input_channels=1,
        n_stages=6,
        features_per_stage=(32, 64, 128, 256, 320, 320),
        conv_op=nn.Conv3d,
        kernel_sizes=None,
        strides=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
        n_blocks_per_stage=(1, 3, 4, 6, 6, 6),
        num_classes=1,
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs=None,
        deep_supervision=False,
        only_last_stage_as_latent=False,
        use_projector=False,
        use_projector_global=True,
        **kwargs,
    ):
        if kernel_sizes is None:
            kernel_sizes = [[3, 3, 3] for _ in range(n_stages)]
        if nonlin_kwargs is None:
            nonlin_kwargs = {"inplace": True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {"eps": 1e-5, "affine": True}

        super().__init__(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
        )

        self.i_adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.v_adaptive_pool = nn.AdaptiveAvgPool3d((16, 16, 16))

        self.use_projector = use_projector
        self.use_projector_global = use_projector_global

        if only_last_stage_as_latent:
            proj_in_dim = features_per_stage[-1]
        else:
            proj_in_dim = sum(features_per_stage)
        self.only_last_stage_as_latent = only_last_stage_as_latent

        if self.use_projector or self.use_projector_global:
            self.projector = nn.Sequential(
                nn.Linear(proj_in_dim, 2048),  # this is technically a linear layer
                nn.BatchNorm1d(2048, affine=False, track_running_stats=False),
                nn.SiLU(),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048, affine=False, track_running_stats=False),
                nn.SiLU(),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048, affine=False, track_running_stats=False),
            )  # output layer

            self.predictor = nn.Sequential(
                nn.Linear(2048, 512),
                nn.BatchNorm1d(512, affine=False, track_running_stats=False),
                nn.SiLU(),
                nn.Linear(512, 2048),
            )

            # initialize the projector weights
            for m in self.projector.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

            for m in self.predictor.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b = x.shape[0]
        skips = self.encoder(x)
        decoded = self.decoder(skips)

        if self.only_last_stage_as_latent:
            skips = [skips[-1]]
        image_latent = torch.concat(
            [self.i_adaptive_pool(s) for s in skips], dim=1
        ).reshape(b, -1)
        patch_latent = torch.concat([self.v_adaptive_pool(s) for s in skips], dim=1)

        if self.use_projector:
            patch_latent = rearrange(patch_latent, "b c w h d -> (b w h d) c")
            patch_latent = self.projector(patch_latent)
            image_latent = self.projector(image_latent)

            if self.training:
                patch_latent = self.predictor(patch_latent)
                image_latent = self.predictor(image_latent)

            patch_latent = rearrange(
                patch_latent, "(b w h d) c -> b c w h d", b=b, w=16, h=16, d=16
            )
        elif self.use_projector_global:
            image_latent = self.projector(image_latent)

            if self.training:
                image_latent = self.predictor(image_latent)
        else:
            patch_latent = patch_latent
            image_latent = image_latent

        return {
            "image_latent": image_latent,
            "proj": patch_latent,
            "recon": decoded,
        }


class FeatureContrastiveDecoderAligned(ConsisMAE):

    def __init__(
        self,
        input_channels=1,
        n_stages=6,
        features_per_stage=(32, 64, 128, 256, 320, 320),
        conv_op=nn.Conv3d,
        kernel_sizes=None,
        strides=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
        n_blocks_per_stage=(1, 3, 4, 6, 6, 6),
        num_classes=1,
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs=None,
        only_last_stage_as_latent=False,
        use_projector=True,
        **kwargs,
    ):
        super().__init__(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=True,
            use_projector=use_projector,
            only_last_stage_as_latent=only_last_stage_as_latent,
            **kwargs,
        )
        self.decoder = UNetDecoder(
            self.encoder,
            num_classes,
            n_conv_per_stage_decoder,
            deep_supervision=True,
            latents=True,
        )

        decoder_proj_in_dim = (
            sum(features_per_stage[:-1])
            if not only_last_stage_as_latent
            else features_per_stage[-1]
        )

        if self.use_projector:
            self.decoder_projector = nn.Sequential(
                nn.Linear(
                    decoder_proj_in_dim, 2048
                ),  # this is technically a linear layer
                nn.BatchNorm1d(2048, affine=False, track_running_stats=False),
                nn.SiLU(),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048, affine=False, track_running_stats=False),
                nn.SiLU(),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048, affine=False, track_running_stats=False),
            )

            self.decoder_predictor = nn.Sequential(
                nn.Linear(2048, 512),
                nn.BatchNorm1d(512, affine=False, track_running_stats=False),
                nn.SiLU(),
                nn.Linear(512, 2048),
            )

            # initialize the projector weights
            for m in self.decoder_projector.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            for m in self.decoder_predictor.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        else:
            self.decoder_projector = None
            self.decoder_predictor = None

    def forward(self, x):
        b = x.shape[0]
        skips = self.encoder(x)
        decodeds, recon = self.decoder(skips)

        if self.only_last_stage_as_latent:
            decodeds = [decodeds[-1]]
            skips = [skips[-1]]

        image_latent = torch.concat(
            [self.i_adaptive_pool(s) for s in skips], dim=1
        ).reshape(b, -1)
        patch_latent = torch.concat([self.v_adaptive_pool(s) for s in decodeds], dim=1)

        patch_latent = rearrange(patch_latent, "b c w h d -> (b w h d) c")

        image_latent = self.projector(image_latent)
        if self.training:
            image_latent = self.predictor(image_latent)

        patch_latent = self.decoder_projector(patch_latent)
        if self.training:
            patch_latent = self.decoder_predictor(patch_latent)

        patch_latent = rearrange(
            patch_latent, "(b w h d) c -> b c w h d", b=b, w=16, h=16, d=16
        )

        return {
            "image_latent": image_latent,
            "proj": patch_latent,
            "recon": recon,
        }


class ConsisEvaMAE(EvaMAE):

    def __init__(
        self,
        input_channels: int,
        embed_dim: int,
        patch_embed_size: Tuple[int, ...],
        output_channels: int,
        input_shape: Tuple[int, int, int] = None,
        use_projector: bool = True,
        use_projector_global: bool = True,
        **kwargs,
    ):
        super().__init__(
            input_channels=input_channels,
            embed_dim=embed_dim,
            patch_embed_size=patch_embed_size,
            output_channels=output_channels,
            input_shape=input_shape,
            **kwargs,
        )

        if not self.use_decoder:
            raise ValueError("ConsisEvaMAE requires a decoder to be used.")

        self.feature_decoder = Eva(
            embed_dim=embed_dim,
            depth=2,  # eva_depth,
            num_heads=16,  # eva_numheads,
            ref_feat_shape=tuple(
                [i // ds for i, ds in zip(input_shape, patch_embed_size)]
            ),
            num_reg_tokens=kwargs.get("num_register_tokens", 0),
            use_rot_pos_emb=kwargs.get("use_rot_pos_emb", True),
            use_abs_pos_emb=kwargs.get("use_abs_pos_emb", True),
            mlp_ratio=kwargs.get("mlp_ratio", 4 * 2 / 3),
            drop_path_rate=kwargs.get("drop_path_rate", 0),
            patch_drop_rate=0,  # No drop in the decoder
            proj_drop_rate=kwargs.get("proj_drop_rate", 0.0),
            attn_drop_rate=kwargs.get("attn_drop_rate", 0.0),
            init_values=kwargs.get("init_values", 0.1),
            scale_attn_inner=kwargs.get("scale_attn_inner", False),
        )

        self.attention_pooling = nn.Linear(embed_dim, 1)

        self.use_projector = use_projector
        self.use_projector_global = use_projector_global

        if self.use_projector or self.use_projector_global:
            self.projector = nn.Sequential(
                nn.Linear(embed_dim, 2048),
                nn.BatchNorm1d(2048),
                nn.SiLU(),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.SiLU(),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
            )  # output layer

            self.predictor = nn.Sequential(
                nn.Linear(2048, 512),
                nn.BatchNorm1d(512),
                nn.SiLU(),
                nn.Linear(512, 2048),
            )

            # initialize the projector weights
            for m in self.projector.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

            for m in self.predictor.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encode patches
        x = self.down_projection(x)
        b, c, w, h, d = x.shape
        x = rearrange(x, "b c w h d -> b (w h d) c")

        # Encode using EVA (internally applies masking with patch_drop_rate)
        encoded, keep_indices = self.eva(x)
        # print(f"Encoded shape: {encoded.shape}, Keep indices shape: {keep_indices.shape if keep_indices is not None else 'None'}")
        num_patches = w * h * d

        if keep_indices is None or not self.training:
            feature_decoded = restored_x = encoded
        else:
            # Restore full sequence with mask tokens
            restored_x = self.restore_full_sequence(encoded, keep_indices, num_patches)
            feature_decoded, _ = self.feature_decoder(restored_x)

        image_latents = torch.matmul(
            torch.softmax(self.attention_pooling(feature_decoded), dim=1).transpose(
                -1, -2
            ),
            feature_decoded,
        ).squeeze(-2)
        patch_latents = feature_decoded

        if self.use_projector:
            patch_latents = rearrange(
                patch_latents, "b (w h d) c -> (b w h d) c", b=b, w=w, h=h, d=d
            )

            patch_latents = self.projector(patch_latents)
            image_latents = self.projector(image_latents)
            if self.training:
                patch_latents = self.predictor(patch_latents)
                image_latents = self.predictor(image_latents)

            patch_latents = rearrange(
                patch_latents, "(b w h d) c -> b c w h d", b=b, w=w, h=h, d=d
            )
        elif self.use_projector_global:
            patch_latents = rearrange(patch_latents, "b (w h d) c -> b c w h d", b=b, w=w, h=h, d=d)
            image_latents = self.projector(image_latents)
            if self.training:
                image_latents = self.predictor(image_latents)

        else:
            # projected = None
            patch_latents = rearrange(patch_latents, "b (w h d) c -> b c w h d", b=b, w=w, h=h, d=d)
            # image_latents = image_latents

        # Decode with restored sequence and rope embeddings
        decoded, _ = self.decoder(restored_x)

        # Project back to output shape
        decoded = rearrange(decoded, "b (w h d) c -> b c w h d", b=b, h=w, w=h, d=d)
        decoded = self.up_projection(decoded)

        return {
            "proj": patch_latents,
            "image_latent": image_latents,
            "recon": decoded,
            "keep_indices": keep_indices,
        }


class FeatureContrastiveDecoderAlignedEva(ConsisEvaMAE):
    """
    This class is a variant of ConsisEvaMAE that uses a decoder with aligned feature contrastive learning.
    It is designed to work with the EVA architecture and includes a feature decoder for aligned feature contrastive learning.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.decoder_projector = nn.Sequential(
            nn.Linear(self.embed_dim, 2048),  # this is technically a linear layer
            nn.BatchNorm1d(2048, affine=False, track_running_stats=False),
            nn.SiLU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048, affine=False, track_running_stats=False),
            nn.SiLU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048, affine=False, track_running_stats=False),
        )

        self.decoder_predictor = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512, affine=False, track_running_stats=False),
            nn.SiLU(),
            nn.Linear(512, 2048),
        )

        # initialize the projector weights
        for m in self.decoder_projector.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for m in self.decoder_predictor.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encode patches
        x = self.down_projection(x)
        b, c, w, h, d = x.shape
        x = rearrange(x, "b c w h d -> b (w h d) c")

        # Encode using EVA (internally applies masking with patch_drop_rate)
        encoded, keep_indices = self.eva(x)
        # print(f"Encoded shape: {encoded.shape}, Keep indices shape: {keep_indices.shape if keep_indices is not None else 'None'}")
        num_patches = w * h * d

        if keep_indices is None or not self.training:
            feature_decoded = restored_x = encoded
        else:
            # Restore full sequence with mask tokens
            restored_x = self.restore_full_sequence(encoded, keep_indices, num_patches)
            feature_decoded, _ = self.feature_decoder(restored_x)

        image_latents = torch.matmul(
            torch.softmax(self.attention_pooling(feature_decoded), dim=1).transpose(
                -1, -2
            ),
            feature_decoded,
        ).squeeze(-2)

        image_latents = self.projector(image_latents)
        if self.training:
            image_latents = self.predictor(image_latents)

        # Decode with restored sequence and rope embeddings
        decoded, _ = self.decoder(restored_x)

        patch_latents = rearrange(
            decoded, "b (w h d) c -> (b w h d) c", b=b, w=w, h=h, d=d
        )
        patch_latents = self.decoder_projector(patch_latents)

        if self.training:
            patch_latents = self.decoder_predictor(patch_latents)

        patch_latents = rearrange(
            patch_latents, "(b w h d) c -> b c w h d", b=b, w=w, h=h, d=d
        )

        # Project back to output shape
        decoded = rearrange(decoded, "b (w h d) c -> b c w h d", b=b, h=w, w=h, d=d)
        decoded = self.up_projection(decoded)

        return {
            "image_latent": image_latents,
            "proj": patch_latents,
            "recon": decoded,
            "keep_indices": keep_indices,
        }


if __name__ == "__main__":
    import os
    import gc
    import psutil

    import thop

    # from nnssl.architectures.architecture_registry import get_res_enc_l

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    def measure_memory(model, input_tensor):
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(input_tensor)
        mem_allocated = torch.cuda.memory_allocated() / (1024**2)  # in MB
        mem_peak = torch.cuda.max_memory_allocated() / (1024**2)  # in MB
        print(f"Current allocated memory: {mem_allocated:.2f} MB")
        print(f"Peak memory usage: {mem_peak:.2f} MB")

    def measure_memory_cpu(model, input_tensor):
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024**2)  # in MB
        with torch.no_grad():
            _ = model(input_tensor)
        mem_after = process.memory_info().rss / (1024**2)  # in MB
        print(f"Memory before: {mem_before:.2f} MB")
        print(f"Memory after: {mem_after:.2f} MB")
        print(f"Memory used by forward pass: {mem_after - mem_before:.2f} MB")

    #
    # Toy example for testing
    input_shape = (64, 64, 64)
    input_tensor = torch.randn(2, 1, *input_shape).to(_device)

    # baseline - get_res_enc_l
    # model = get_res_enc_l(
    #     num_input_channels=1,
    #     num_output_channels=1,
    #     deep_supervision=False,
    # )
    model = ConsisMAE()
    model = model.to(_device)
    # make the decoder an identity function
    model.decoder = nn.Identity()
    model.train(False)
    if _device == "cuda":
        measure_memory(model, input_tensor)
    else:
        measure_memory_cpu(model, input_tensor)
    flops, params = thop.profile(model, inputs=(input_tensor,), verbose=False)
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Parameters: {params / 1e6:.2f} M")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # # FeatureContrastiveDecoderAligned - CNN
    # model = FeatureContrastiveDecoderAligned(
    #     input_channels=1,
    #     num_classes=1,
    # )
    # model = model.to(_device)
    # model.train(True)
    # if _device == "cuda":
    #     measure_memory(model, input_tensor)
    # else:
    #     measure_memory_cpu(model, input_tensor)
    # flops, params = thop.profile(model, inputs=(input_tensor,), verbose=False)
    # print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    # print(f"Parameters: {params / 1e6:.2f} M")
    # del model
    # gc.collect()
    # torch.cuda.empty_cache()
    #
    # # FeatureContrastiveDecoderAlignedEva - EVA
    # model = FeatureContrastiveDecoderAlignedEva(
    #     patch_embed_size=(8, 8, 8),
    #     input_channels=1,
    #     embed_dim=192,
    #     output_channels=1,
    #     input_shape=input_shape,
    #     decoder_eva_depth=6,
    #     decoder_eva_numheads=8,
    #     patch_drop_rate=0.7,
    # )
    # model = model.to(_device)
    # model.train(True)
    # if _device == "cuda":
    #     measure_memory(model, input_tensor)
    # else:
    #     measure_memory_cpu(model, input_tensor)
    # flops, params = thop.profile(model, inputs=(input_tensor,), verbose=False)
    # print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    # print(f"Parameters: {params / 1e6:.2f} M")
    # del model
    # gc.collect()
    # torch.cuda.empty_cache()
