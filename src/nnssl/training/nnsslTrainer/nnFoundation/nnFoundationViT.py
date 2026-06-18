import torch
from torch import nn

from batchgenerators.utilities.file_and_folder_operations import save_json

from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans
from nnssl.architectures.evaMAE_module import EvaMAE
from nnssl.experiment_planning.experiment_planners.plan import Plan
from nnssl.training.nnsslTrainer.masked_image_modeling.BaseEvaMAETrainer import BaseEvaMAETrainer


class BaseEvaMAETrainerPrimusX(BaseEvaMAETrainer):
    def __init__(
            self,
            plan: Plan,
            configuration_name: str,
            fold: int,
            pretrain_json: dict,
            device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.config_plan.patch_size = (192, 192, 192)
        self.recommended_downstream_patchsize = (192, 192, 192)
        self.mask_percentage = 0.8

    def build_architecture_and_adaptation_plan(
            self, config_plan, num_input_channels, num_output_channels) -> nn.Module:
        network = EvaMAE(
            input_channels=1,
            embed_dim=self.embed_dim,
            patch_embed_size=self.vit_patch_size,
            output_channels=1,
            input_shape=tuple(self.config_plan.patch_size),
            encoder_eva_depth=self.encoder_eva_depth,
            encoder_eva_numheads=self.encoder_eva_numheads,
            decoder_eva_depth=self.decoder_eva_depth,
            decoder_eva_numheads=self.decoder_eva_numheads,
            patch_drop_rate=self.mask_percentage,
            drop_path_rate=self.drop_path_rate,
            attn_drop_rate=self.attention_drop_rate,
            init_values=self.init_value,
            scale_attn_inner=self.scale_attn_inner,)

        arch_dict = {"input_channels": 1,
                     "embed_dim": self.embed_dim,
                     "patch_embed_size": self.vit_patch_size,
                     "output_channels": 1,
                     "input_shape": tuple(self.config_plan.patch_size),
                     "encoder_eva_depth": self.encoder_eva_depth,
                     "encoder_eva_numheads": self.encoder_eva_numheads,
                     "decoder_eva_depth": self.decoder_eva_depth,
                     "decoder_eva_numheads": self.decoder_eva_numheads,
                     "patch_drop_rate": self.mask_percentage,
                     "drop_path_rate": self.drop_path_rate,
                     "attn_drop_rate": self.attention_drop_rate,
                     "init_values": self.init_value,
                     "scale_attn_inner": self.scale_attn_inner}

        adapt_plan = AdaptationPlan(architecture_plans=ArchitecturePlans("PrimusX", arch_kwargs=arch_dict),
                                    pretrain_plan=self.plan,
                                    pretrain_num_input_channels=1,
                                    recommended_downstream_patchsize=self.recommended_downstream_patchsize,
                                    key_to_encoder="eva",
                                    key_to_stem="down_projection",
                                    keys_to_in_proj=("down_projection.proj",),
                                    key_to_lpe="eva.pos_embed",
                                    )
        save_json(adapt_plan.serialize(), self.adaptation_json_plan)
        n_params = sum(p.numel() for p in network.parameters())
        self.print_to_log_file(f"Number of parameters: {n_params}")

        return network, adapt_plan


class BaseEvaMAETrainer_BS96_192ps_2500ep_40_16_8_16_1056_lr2e3(BaseEvaMAETrainerPrimusX):
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
        self.num_epochs = 2500
        self.initial_lr = 2e-3
        self.embed_dim = 1056
        self.encoder_eva_depth = 40
        self.encoder_eva_numheads = 16
        self.decoder_eva_depth = 8
        self.decoder_eva_numheads = 16
        self.init_value = 0.1
        self.scale_attn_inner = True


class BaseEvaMAETrainer_BS96_192ps_2500ep_200wu_16_12_4_12_864_lr1e4_nolayerscale(BaseEvaMAETrainerPrimusX):
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
        self.num_epochs = 2500
        self.initial_lr = 1e-4
        self.warmup_duration_whole_net = 200
        self.embed_dim = 864
        self.encoder_eva_depth = 16
        self.encoder_eva_numheads = 12
        self.decoder_eva_depth = 4
        self.decoder_eva_numheads = 12
        self.init_value = None
        self.scale_attn_inner = False
