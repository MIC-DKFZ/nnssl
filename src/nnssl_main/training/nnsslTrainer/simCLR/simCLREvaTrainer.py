from copy import deepcopy
from typing import Union, Tuple, List

import numpy as np
import torch

from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans
from nnssl.training.lr_scheduler.warmup import Lin_incr_LRScheduler, PolyLRScheduler_offset
from nnssl.architectures.evaMAE_module import EvaMAE
from nnssl.training.nnsslTrainer.simCLR.simCLRTrainer import SimCLRTrainer
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from torch import autocast
from nnssl.architectures.voco_architecture import VoCoEvaArchitecture
from nnssl.utilities.helpers import dummy_context, empty_cache

from nnssl.experiment_planning.experiment_planners.plan import Plan


class SimCLREvaTrainer(SimCLRTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
        patch_size: tuple = (192, 192, 64),
        crop_size: tuple = (64, 64, 64),
        num_crops_per_image: int = 2,
        min_crop_overlap: float = 0.5,
    ):
        super().__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
            patch_size,
            crop_size,
            num_crops_per_image,
            min_crop_overlap,
        )

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
            input_shape=tuple(self.crop_size),
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
        architecture = VoCoEvaArchitecture(encoder, self.embed_dim)

        plan = deepcopy(self.plan)
        plan.configurations[self.configuration_name].patch_size = self.crop_size

        adapt_plan = AdaptationPlan(
            architecture_plans=ArchitecturePlans("PrimusM"),
            pretrain_plan=plan,
            recommended_downstream_patchsize=self.recommended_downstream_patchsize,
            pretrain_num_input_channels=1,  # This is the actual input patch size!
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

    def train_step(self, batch: Tuple[dict, dict]) -> dict:

        all_crops = batch["all_crops"]
        NREF = batch["reference_crop_index"]

        all_crops = all_crops.to(self.device, non_blocking=True)

        if torch.isnan(all_crops).any():
            print("NaN values found in input data!")
        if torch.isinf(all_crops).any():
            print("Infinity values found in input data!")

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            all_crop_embeddings = self.network(all_crops)
            if torch.isnan(all_crop_embeddings).any():
                print("NaN values found in embeddings!")

            # This rearrange below isn't necessary, but would come in handy when doing more involved contrastive tasks.
            # z_i_embeddings = rearrange(
            #     all_crop_embeddings[:NREF], "(b NREF) c -> b NREF c", b=self.batch_size
            # )
            # z_j_embeddings = rearrange(
            #     all_crop_embeddings[NREF:], "(b NREF) c -> b NREF c", b=self.batch_size
            # )

            # Normalize prior to contrastive loss
            z_i_embeddings = nn.functional.normalize(all_crop_embeddings[:NREF], dim=1)
            z_j_embeddings = nn.functional.normalize(all_crop_embeddings[NREF:], dim=1)

            # del data
            l, acc = self.loss(z_i_embeddings, z_j_embeddings)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            # for name, param in self.network.named_parameters():
            #     if param.grad is not None:
            #         if param.grad.norm() > 1:
            #             print(f"{name}: gradient norm: {param.grad.norm()}")
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
            self.optimizer.step()

        # print(f"Train loss: {l.detach().cpu().numpy()} - accuracy: {acc}")

        return {"loss": l.detach().cpu().numpy()}


####################################################################
############################# VARIANTS #############################
####################################################################


class SimCLREvaTrainer_BS2(SimCLREvaTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 2


class SimCLREvaTrainer_BS32(SimCLREvaTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 32
