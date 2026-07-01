from typing import Union

import torch
from torch import nn, autocast
from torch._dynamo import OptimizedModule

from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans
from nnssl.architectures.swinunetr_architecture import SwinUNETREvaArchitecture
from nnssl.experiment_planning.experiment_planners.plan import Plan

from torch.nn.parallel import DistributedDataParallel as DDP

from nnssl.training.nnsslTrainer.swinunetr_pretrain.SwinUNETRTrainer import SwinUNETRTrainer
from nnssl.utilities.helpers import dummy_context, empty_cache
from nnssl.training.lr_scheduler.warmup import Lin_incr_LRScheduler, PolyLRScheduler_offset
from nnssl.architectures.evaMAE_module import EvaMAE
from batchgenerators.utilities.file_and_folder_operations import save_json


class SwinUNETREvaTrainer(SwinUNETRTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.config_plan.patch_size = (160, 160, 160)
        self.rec_loss_weight = 1
        self.contrast_loss_weight = 1
        self.rot_loss_weight = 1

        ###settings taken from fabi
        self.drop_path_rate = 0.2
        self.attention_drop_rate = 0
        self.grad_clip = 1
        self.initial_lr = 3e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = False
        self.warmup_duration_whole_net = 50  # lin increase whole network
        self.training_stage = None

        self.mask_ratio = 0.0
        self.vit_patch_size = (8, 8, 8)
        self.embed_dim = 864
        self.encoder_eva_depth = 16
        self.encoder_eva_numheads = 12
        self.init_value = 0.1
        self.scale_attn_inner = True

    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("warmup_all")
        elif self.current_epoch == self.warmup_duration_whole_net:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("train")

        super().on_train_epoch_start()

    def configure_optimizers(self, stage: str = "warmup_all"):
        assert stage in ["warmup_all", "train"]

        if self.training_stage == stage:
            return self.optimizer, self.lr_scheduler

        if isinstance(self.network, DDP):
            params = self.network.module.parameters()
        else:
            params = self.network.parameters()

        if stage == "warmup_all":
            self.print_to_log_file("train whole net, warmup")
            optimizer = torch.optim.AdamW(
                params, self.initial_lr, weight_decay=self.weight_decay, amsgrad=False, betas=(0.9, 0.98), fused=True
            )
            lr_scheduler = Lin_incr_LRScheduler(optimizer, self.initial_lr, self.warmup_duration_whole_net)
            self.print_to_log_file(f"Initialized warmup_all optimizer and lr_scheduler at epoch {self.current_epoch}")
        else:
            self.print_to_log_file("train whole net, default schedule")
            if self.training_stage == "warmup_all":
                # we can keep the existing optimizer and don't need to create a new one. This will allow us to keep
                # the accumulated momentum terms which already point in a useful driection
                optimizer = self.optimizer
            else:
                optimizer = torch.optim.AdamW(
                    params,
                    self.initial_lr,
                    weight_decay=self.weight_decay,
                    amsgrad=False,
                    betas=(0.9, 0.98),
                    fused=True,
                )
            lr_scheduler = PolyLRScheduler_offset(
                optimizer, self.initial_lr, self.num_epochs, self.warmup_duration_whole_net
            )
            self.print_to_log_file(f"Initialized train optimizer and lr_scheduler at epoch {self.current_epoch}")
        self.training_stage = stage
        empty_cache(self.device)
        return optimizer, lr_scheduler

    def build_architecture_and_adaptation_plan(
        self, config_plan, num_input_channels, num_output_channels
    ) -> nn.Module:
        encoder = EvaMAE(
            input_channels=1,
            embed_dim=self.embed_dim,
            patch_embed_size=self.vit_patch_size,
            output_channels=num_output_channels,
            input_shape=tuple(self.config_plan.patch_size),
            encoder_eva_depth=self.encoder_eva_depth,
            encoder_eva_numheads=self.encoder_eva_numheads,
            decoder_eva_depth=0,
            decoder_eva_numheads=0,
            patch_drop_rate=self.mask_ratio,
            drop_path_rate=self.drop_path_rate,
            attn_drop_rate=self.attention_drop_rate,
            init_values=self.init_value,
            scale_attn_inner=self.scale_attn_inner,
            do_up_projection=False,
        )
        architecture = SwinUNETREvaArchitecture(encoder, self.num_output_channels)
        adapt_plan = AdaptationPlan(
            architecture_plans=ArchitecturePlans("PrimusM"),
            pretrain_plan=self.plan,
            recommended_downstream_patchsize=self.recommended_downstream_patchsize,
            pretrain_num_input_channels=1,
            key_to_encoder="encoder.eva",
            key_to_stem="encoder.down_projection",
            key_to_in_proj=("encoder.down_projection.proj",),
            key_to_lpe="encoder.eva.pos_embed",
        )
        return architecture, adapt_plan

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint["network_weights"].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith("module."):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint["init_args"]

        self.current_epoch = checkpoint["current_epoch"]
        min_epoch = self.logger.load_checkpoint(checkpoint["logging"])
        # Apparently the val log is not written correctly when we currently save the checkpoint.
        self.current_epoch = min_epoch
        self._best_ema = checkpoint["_best_ema"]

        # messing with state dict naming schemes. Facepalm.
        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)

        # it's fine to do this every time we load because configure_optimizers will be a no-op if the correct optimizer
        # and lr scheduler are already set up
        if self.current_epoch < self.warmup_duration_whole_net:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("warmup_all")
        else:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("train")

        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.grad_scaler is not None:
            if checkpoint["grad_scaler_state"] is not None:
                self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state"])

    def train_step(self, batch: dict) -> dict:

        imgs1_rotated, imgs2_rotated = batch["imgs_rotated"]
        rotations1, rotations2 = batch["rotations"]
        imgs1_rotated_cutout, imgs2_rotated_cutout = batch["imgs_rotated_cutout"]
        # print(f"rank: {self.local_rank}", imgs1_rotated_cutout.shape)

        imgs_rotated = torch.cat([imgs1_rotated, imgs2_rotated], dim=0)
        rotations = torch.cat([rotations1, rotations2], dim=0)
        imgs_rotated_cutout = torch.cat([imgs1_rotated_cutout, imgs2_rotated_cutout], dim=0)

        imgs_rotated = imgs_rotated.to(self.device, non_blocking=True)
        rotations = rotations.to(self.device, non_blocking=True)
        imgs_rotated_cutout = imgs_rotated_cutout.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            rotations_pred, contrast_pred, reconstructions = self.network(imgs_rotated_cutout)
            l = self.loss(rotations_pred, rotations, contrast_pred, reconstructions, imgs_rotated)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}


class SwinUNETREvaTrainer_BS8(SwinUNETREvaTrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.config_plan.patch_size = (160, 160, 160)
        self.total_batch_size = 8


class SwinUNETREvaTrainer_test(SwinUNETREvaTrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.config_plan.patch_size = (64, 64, 64)
        self.total_batch_size = 4
