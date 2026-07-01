"""
BaseDINOv3ViTTrainer
====================
nnSSL trainer with:
  - 3D ViT-Base encoder (from cardiac DINO pipeline)
  - DINOv3 losses: DINO + iBOT + KoLeo

vs BaseDINOv3UxLSTMTrainer:
  Backbone: UxLSTMBot → 3D ViT-Base
  Everything else identical (losses, EMA, heads)

Usage:
    nnssl_train Dataset910_Combined onemmiso -tr BaseDINOv3ViTTrainer_BS1_patch128

Author: Abdul Qayyum
"""

import os, sys, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from typing_extensions import override
from torch import autocast

# Add cardiac DINO to path
DINOV2_DIR = '/mnt/all_data/Abdul/Heart_vjepa2/Heart_dinov3'
sys.path.insert(0, DINOV2_DIR)

from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import BaseMAETrainer
from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from batchgenerators.utilities.file_and_folder_operations import save_json


# ── DINOv3 Losses (same as UxLSTM trainer) ───────────────────────────────────

class DINOLoss(nn.Module):
    def __init__(self, n_proto=65536, s_temp=0.1, t_temp=0.04, center_mom=0.9):
        super().__init__()
        self.s_temp = s_temp
        self.t_temp = t_temp
        self.center_mom = center_mom
        self.register_buffer("center", torch.zeros(1, n_proto))

    @torch.no_grad()
    def update_center(self, t):
        self.center = self.center * self.center_mom + t.mean(0, keepdim=True) * (1 - self.center_mom)

    def forward(self, s, t):
        s_p = F.softmax(s / self.s_temp, dim=-1)
        t_p = F.softmax((t - self.center) / self.t_temp, dim=-1).detach()
        loss = -(t_p * torch.log(s_p + 1e-8)).sum(-1).mean()
        self.update_center(t)
        return loss


class iBOTLoss(nn.Module):
    def __init__(self, n_proto=65536, s_temp=0.1, t_temp=0.04, center_mom=0.9):
        super().__init__()
        self.s_temp = s_temp
        self.t_temp = t_temp
        self.c_mom = center_mom
        self.register_buffer("center", torch.zeros(1, 1, n_proto))

    @torch.no_grad()
    def update_center(self, t):
        self.center = self.center * self.c_mom + t.mean((0,1), keepdim=True) * (1 - self.c_mom)

    def forward(self, s, t, mask=None):
        s_p = F.softmax(s / self.s_temp, dim=-1)
        t_p = F.softmax((t - self.center) / self.t_temp, dim=-1).detach()
        loss = -(t_p * torch.log(s_p + 1e-8)).sum(-1)
        loss = (loss * mask.float()).sum() / (mask.float().sum() + 1e-8) if mask is not None else loss.mean()
        self.update_center(t)
        return loss


class KoLeoLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def forward(self, x, eps=1e-3):
        with torch.autocast("cuda", enabled=False):
            x = F.normalize(x.float(), p=2, dim=-1)
            dots = torch.mm(x, x.t())
            dots.view(-1)[:: (x.shape[0] + 1)].fill_(-1)
            _, idx = torch.max(dots, dim=1)
            dist = self.pdist(x, x[idx]).clamp(min=eps)
            loss = -torch.log(dist).mean().clamp(max=100.0)
        return loss


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=65536, hidden=2048, bottleneck=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, bottleneck),
        )
        self.last = nn.Linear(bottleneck, out_dim, bias=False)
        nn.init.normal_(self.last.weight, 0, 0.01)

    def forward(self, x):
        x = F.normalize(self.mlp(x), dim=-1)
        w = F.normalize(self.last.weight, dim=1)
        return F.linear(x, w)


@torch.no_grad()
def ema_update(student, teacher, m):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(m).add_(ps.data, alpha=1 - m)


# ── 3D ViT Feature Extractor ──────────────────────────────────────────────────

class ViT3DFeatureExtractor(nn.Module):
    """
    Wraps our 3D ViT-Base to extract CLS + patch tokens.
    Input:  (B, C, D, H, W) — nnSSL format
    Output: cls (B, 768), patches (B, N, 768)
    """
    def __init__(self, img_size=(128,128,128), patch_size=(16,16,16),
                 in_chans=1, embed_dim=768, num_register_tokens=4):
        super().__init__()
        from models.vision_transformer_3d import vit_base
        self.encoder = vit_base(
            patch_size=patch_size,
            img_size=img_size,
            in_chans=in_chans,
            num_register_tokens=num_register_tokens,
        )
        self.embed_dim = embed_dim
        self.n_reg = num_register_tokens

    def forward(self, x):
        enc = self.encoder
        B   = x.shape[0]

        # Patch embed
        tokens = enc.patch_embed(x)
        cls    = enc.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + enc.interpolate_pos_encoding(
            tokens, x.shape[2], x.shape[3], x.shape[4])

        # Register tokens
        if enc.register_tokens is not None:
            reg    = enc.register_tokens.expand(B, -1, -1)
            tokens = torch.cat([tokens[:,:1], reg, tokens[:,1:]], dim=1)

        n_prefix = 1 + self.n_reg

        # All 12 blocks
        if enc.chunked_blocks:
            all_blocks = [b for chunk in enc.blocks for b in chunk]
        else:
            all_blocks = list(enc.blocks)

        for blk in all_blocks:
            tokens = blk(tokens)

        tokens = enc.norm(tokens)

        cls_token    = tokens[:, 0]              # (B, 768)
        patch_tokens = tokens[:, n_prefix:]      # (B, N, 768)

        return cls_token, patch_tokens


# ── Main Trainer ──────────────────────────────────────────────────────────────

class BaseDINOv3ViTTrainer(BaseMAETrainer):
    """
    nnSSL DINOv3 trainer with 3D ViT-Base backbone.
    
    Key differences from BaseDINOv3UxLSTMTrainer:
      Backbone: UxLSTMBot → 3D ViT-Base (768-dim)
      Patch:    CNN stages → ViT patch embedding (16³)
      
    Same losses: DINO + iBOT + KoLeo
    Same EMA teacher/student framework
    """

    def __init__(self, plan: Plan, configuration_name: str, fold: int,
                 pretrain_json: dict, device=torch.device("cuda")):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        # DINOv3 config
        self.n_proto      = 65536
        self.s_temp       = 0.1
        self.t_temp_init  = 0.04
        self.t_temp_final = 0.04
        self.t_warmup_ep  = 10
        self.center_mom   = 0.9
        self.ema_start    = 0.996
        self.ema_end      = 1.0
        self.lambda_dino  = 1.0
        self.lambda_ibot  = 1.0
        self.lambda_koleo = 0.01
        self.mask_ratio   = 0.5
        self.global_step  = 0
        self.grad_accum_steps = 4  # effective BS=4
        self.embed_dim    = 768  # ViT-Base
        self.patch_size   = (16, 16, 16)  # isotropic for CT

    @override
    def build_architecture_and_adaptation_plan(
        self, config_plan, num_input_channels, num_output_channels, *args, **kwargs
    ) -> Tuple[nn.Module, AdaptationPlan]:
        """
        Build 3D ViT-Base as the network.
        ViT3DFeatureExtractor is built in initialize() for student/teacher.
        This returns the backbone for nnSSL compatibility.
        """
        patch_size = tuple(config_plan.patch_size)

        # Build ViT3D as main network
        architecture = ViT3DFeatureExtractor(
            img_size=patch_size,
            patch_size=self.patch_size,
            in_chans=num_input_channels,
            embed_dim=self.embed_dim,
            num_register_tokens=4,
        )

        arch_plans = ArchitecturePlans(arch_class_name="ResEncL")  # ViT3D not registered — use ResEncL as placeholder
        adapt_plan  = AdaptationPlan(
            architecture_plans=arch_plans,
            pretrain_plan=self.plan,
            pretrain_num_input_channels=num_input_channels,
            recommended_downstream_patchsize=config_plan.patch_size,
            key_to_encoder="encoder",
            key_to_stem="encoder.patch_embed",
            keys_to_in_proj=("encoder.patch_embed.proj",),
        )
        save_json(adapt_plan.serialize(), self.adaptation_json_plan)
        return architecture, adapt_plan

    def initialize(self):
        super().initialize()

        patch_size = tuple(self.config_plan.patch_size)

        # Build ViT feature extractor
        self.feature_extractor = ViT3DFeatureExtractor(
            img_size=patch_size,
            patch_size=self.patch_size,
            in_chans=1,
            embed_dim=self.embed_dim,
            num_register_tokens=4,
        ).to(self.device)

        # Teacher = EMA copy
        self.teacher_extractor = copy.deepcopy(self.feature_extractor)
        for p in self.teacher_extractor.parameters():
            p.requires_grad_(False)

        # Projection heads
        self.s_cls_head   = DINOHead(self.embed_dim, self.n_proto).to(self.device)
        self.t_cls_head   = copy.deepcopy(self.s_cls_head)
        self.s_patch_head = DINOHead(self.embed_dim, self.n_proto).to(self.device)
        self.t_patch_head = copy.deepcopy(self.s_patch_head)

        for p in self.t_cls_head.parameters():   p.requires_grad_(False)
        for p in self.t_patch_head.parameters(): p.requires_grad_(False)

        # Losses
        self.dino_loss  = DINOLoss(self.n_proto, self.s_temp,
                                    self.t_temp_final, self.center_mom).to(self.device)
        self.ibot_loss  = iBOTLoss(self.n_proto, self.s_temp,
                                    self.t_temp_final, self.center_mom).to(self.device)
        self.koleo_loss = KoLeoLoss().to(self.device)

        # Optimizer — add heads
        self.optimizer.add_param_group({
            'params': list(self.feature_extractor.parameters()) +
                      list(self.s_cls_head.parameters()) +
                      list(self.s_patch_head.parameters()),
            'lr': self.initial_lr
        })

    def verify_adaptation_plans(self, *args, **kwargs):
        """Skip verification — ViT3D not registered in nnSSL architecture registry."""
        self.print_to_log_file("[DINOv3-ViT] Skipping adaptation plan verification for ViT3D.")

    def _get_teacher_temp(self):
        ep = self.current_epoch
        if ep < self.t_warmup_ep:
            return self.t_temp_init + (self.t_temp_final - self.t_temp_init) * ep / self.t_warmup_ep
        return self.t_temp_final

    def _get_ema_momentum(self):
        total = self.num_epochs * 250
        return self.ema_end - (self.ema_end - self.ema_start) * \
               (np.cos(np.pi * self.global_step / total) + 1) / 2

    def _forward_dinov3(self, data):
        """Shared forward for train and validation."""
        t_temp = self._get_teacher_temp()
        self.dino_loss.t_temp = t_temp
        self.ibot_loss.t_temp = t_temp

        with autocast(self.device.type, enabled=True):
            s_cls, s_patches = self.feature_extractor(data)
            with torch.no_grad():
                t_cls, t_patches = self.teacher_extractor(data)

            s_cls_p   = self.s_cls_head(s_cls)
            t_cls_p   = self.t_cls_head(t_cls)
            s_patch_p = self.s_patch_head(s_patches)
            t_patch_p = self.t_patch_head(t_patches)

            B, N, _ = s_patches.shape
            mask = (torch.rand(B, N, device=self.device) < self.mask_ratio)

            l_dino  = self.dino_loss(s_cls_p, t_cls_p)
            l_ibot  = self.ibot_loss(s_patch_p, t_patch_p, mask)
            l_koleo = self.koleo_loss(s_cls)
            loss    = (self.lambda_dino  * l_dino  +
                       self.lambda_ibot  * l_ibot  +
                       self.lambda_koleo * l_koleo)
        return loss, l_dino, l_ibot, l_koleo, s_cls

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        if isinstance(data, (list, tuple)):
            data = data[0]
        data = data.to(self.device, non_blocking=True)

        loss, l_dino, l_ibot, l_koleo, s_cls = self._forward_dinov3(data)

        if not torch.isfinite(loss):
            del data
            return {"loss": 0.0}

        self.optimizer.zero_grad(set_to_none=True)
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(
            list(self.feature_extractor.parameters()) +
            list(self.s_cls_head.parameters()) +
            list(self.s_patch_head.parameters()), 1.0)
        if torch.isfinite(total_norm):
            self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        m = self._get_ema_momentum()
        ema_update(self.feature_extractor, self.teacher_extractor, m)
        ema_update(self.s_cls_head,   self.t_cls_head,   m)
        ema_update(self.s_patch_head, self.t_patch_head, m)

        self.global_step += 1
        del data
        return {"loss": loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        if isinstance(data, (list, tuple)):
            data = data[0]
        data = data.to(self.device, non_blocking=True)

        with torch.no_grad():
            loss, l_dino, l_ibot, l_koleo, _ = self._forward_dinov3(data)

        del data
        return {"loss": loss.detach().cpu().numpy()}

    def save_checkpoint(self, filename):
        super().save_checkpoint(filename)
        # Save teacher
        teacher_path = filename.replace(".pth", "_teacher.pth")
        torch.save({
            "teacher":      self.teacher_extractor.state_dict(),
            "t_cls_head":   self.t_cls_head.state_dict(),
            "t_patch_head": self.t_patch_head.state_dict(),
            "epoch":        self.current_epoch,
        }, teacher_path)
        # Save student ViT feature extractor separately for LP/EAO
        vit_path = filename.replace(".pth", "_vit_student.pth")
        torch.save({
            "feature_extractor": self.feature_extractor.state_dict(),
            "s_cls_head":        self.s_cls_head.state_dict(),
            "s_patch_head":      self.s_patch_head.state_dict(),
            "embed_dim":         self.embed_dim,
            "patch_size":        self.patch_size,
            "epoch":             self.current_epoch,
        }, vit_path)

    def run_online_evaluation(self, *args, **kwargs):
        pass

    def finish_online_evaluation(self):
        if self.online_eval_losses:
            mean = np.mean(self.online_eval_losses)
            self.print_to_log_file(f"[DINOv3-ViT] Mean loss: {mean:.4f}")
            self.online_eval_losses = []




# Removed KoLeo completely from ViT trainer

class BaseDINOv3ViTTrainer_NoKoLeo(BaseMAETrainer):
    def __init__(self, plan, configuration_name, fold, pretrain_json, device=torch.device("cuda")):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)

        self.n_proto      = 65536
        self.s_temp       = 0.1
        self.t_temp_init  = 0.04
        self.t_temp_final = 0.04
        self.t_warmup_ep  = 10
        self.center_mom   = 0.9
        self.ema_start    = 0.996
        self.ema_end      = 1.0

        self.lambda_dino  = 1.0
        self.lambda_ibot  = 1.0

        self.mask_ratio   = 0.5
        self.global_step  = 0
        self.grad_accum_steps = 4
        self.embed_dim    = 768
        self.patch_size   = (16, 16, 16)

    def initialize(self):
        super().initialize()

        patch_size = tuple(self.config_plan.patch_size)

        self.feature_extractor = ViT3DFeatureExtractor(
            img_size=patch_size,
            patch_size=self.patch_size,
            in_chans=1,
            embed_dim=self.embed_dim,
            num_register_tokens=4,
        ).to(self.device)

        self.teacher_extractor = copy.deepcopy(self.feature_extractor)
        for p in self.teacher_extractor.parameters():
            p.requires_grad_(False)

        self.s_cls_head   = DINOHead(self.embed_dim, self.n_proto).to(self.device)
        self.t_cls_head   = copy.deepcopy(self.s_cls_head)
        self.s_patch_head = DINOHead(self.embed_dim, self.n_proto).to(self.device)
        self.t_patch_head = copy.deepcopy(self.s_patch_head)

        for p in self.t_cls_head.parameters():   p.requires_grad_(False)
        for p in self.t_patch_head.parameters(): p.requires_grad_(False)

        self.dino_loss  = DINOLoss(self.n_proto, self.s_temp,
                                  self.t_temp_final, self.center_mom).to(self.device)
        self.ibot_loss  = iBOTLoss(self.n_proto, self.s_temp,
                                  self.t_temp_final, self.center_mom).to(self.device)

        self.optimizer.add_param_group({
            'params': list(self.feature_extractor.parameters()) +
                      list(self.s_cls_head.parameters()) +
                      list(self.s_patch_head.parameters()),
            'lr': self.initial_lr
        })

    def _get_teacher_temp(self):
        ep = self.current_epoch
        if ep < self.t_warmup_ep:
            return self.t_temp_init + (self.t_temp_final - self.t_temp_init) * ep / self.t_warmup_ep
        return self.t_temp_final

    def _get_ema_momentum(self):
        total = self.num_epochs * 250
        return self.ema_end - (self.ema_end - self.ema_start) * \
               (np.cos(np.pi * self.global_step / total) + 1) / 2

    def _forward_dinov3(self, data):
        t_temp = self._get_teacher_temp()
        self.dino_loss.t_temp = t_temp
        self.ibot_loss.t_temp = t_temp

        with autocast(self.device.type, enabled=True):
            s_cls, s_patches = self.feature_extractor(data)
            with torch.no_grad():
                t_cls, t_patches = self.teacher_extractor(data)

            s_cls_p   = self.s_cls_head(s_cls)
            t_cls_p   = self.t_cls_head(t_cls)
            s_patch_p = self.s_patch_head(s_patches)
            t_patch_p = self.t_patch_head(t_patches)

            B, N, _ = s_patches.shape
            mask = (torch.rand(B, N, device=self.device) < self.mask_ratio)

            l_dino = self.dino_loss(s_cls_p, t_cls_p)
            l_ibot = self.ibot_loss(s_patch_p, t_patch_p, mask)

            loss = self.lambda_dino * l_dino + self.lambda_ibot * l_ibot

        return loss, l_dino, l_ibot

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        if isinstance(data, (list, tuple)):
            data = data[0]
        data = data.to(self.device, non_blocking=True)

        loss, l_dino, l_ibot = self._forward_dinov3(data)

        self.optimizer.zero_grad(set_to_none=True)
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)

        torch.nn.utils.clip_grad_norm_(
            list(self.feature_extractor.parameters()) +
            list(self.s_cls_head.parameters()) +
            list(self.s_patch_head.parameters()), 1.0)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        m = self._get_ema_momentum()
        ema_update(self.feature_extractor, self.teacher_extractor, m)
        ema_update(self.s_cls_head,   self.t_cls_head,   m)
        ema_update(self.s_patch_head, self.t_patch_head, m)

        self.global_step += 1
        del data
        return {"loss": loss.detach().cpu().numpy()}

    def save_checkpoint(self, filename):
        super().save_checkpoint(filename)
        # Save teacher extractor
        teacher_path = filename.replace(".pth", "_teacher.pth")
        torch.save({
            "teacher":           self.teacher_extractor.state_dict(),
            "t_cls_head":        self.t_cls_head.state_dict(),
            "t_patch_head":      self.t_patch_head.state_dict(),
            "epoch":             self.current_epoch,
        }, teacher_path)
        # Save student ViT feature extractor for LP/EAO downstream
        vit_path = filename.replace(".pth", "_vit_student.pth")
        torch.save({
            "feature_extractor": self.feature_extractor.state_dict(),
            "s_cls_head":        self.s_cls_head.state_dict(),
            "s_patch_head":      self.s_patch_head.state_dict(),
            "embed_dim":         self.embed_dim,
            "patch_size":        self.patch_size,
            "epoch":             self.current_epoch,
        }, vit_path)

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        if isinstance(data, (list, tuple)):
            data = data[0]
        data = data.to(self.device, non_blocking=True)

        with torch.no_grad():
            loss, l_dino, l_ibot = self._forward_dinov3(data)

        del data
        return {"loss": loss.detach().cpu().numpy()}


# Variant matching your run
class BaseDINOv3ViTTrainer_NoKoLeo_BS1_patch128_ep200(BaseDINOv3ViTTrainer_NoKoLeo):
    def __init__(self, plan, configuration_name, fold, pretrain_json, device=torch.device("cuda")):
        plan.configurations[configuration_name].batch_size = 1
        plan.configurations[configuration_name].patch_size = (128, 128, 128)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)

        self.config_plan.patch_size = (128, 128, 128)
        self.batch_size = 1
        self.num_epochs = 200





# ── Variants ──────────────────────────────────────────────────────────────────

class BaseDINOv3ViTTrainer_BS1_patch128(BaseDINOv3ViTTrainer):
    """BS=1, patch 128³ — isotropic CT patches, patch_embed 16³."""
    def __init__(self, plan, configuration_name, fold, pretrain_json,
                 device=torch.device("cuda")):
        plan.configurations[configuration_name].batch_size = 1
        plan.configurations[configuration_name].patch_size = (128, 128, 128)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.config_plan.patch_size = (128, 128, 128)
        self.batch_size = 1


class BaseDINOv3ViTTrainer_BS1_patch128_ep200(BaseDINOv3ViTTrainer_BS1_patch128):
    """BS=1, patch 128³, 200 epochs."""
    def __init__(self, plan, configuration_name, fold, pretrain_json,
                 device=torch.device("cuda")):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.num_epochs = 200


class BaseDINOv3ViTTrainer_BS2_patch96_ep200(BaseDINOv3ViTTrainer):
    """BS=2, patch 96³ — smaller patch fits more in memory."""
    def __init__(self, plan, configuration_name, fold, pretrain_json,
                 device=torch.device("cuda")):
        plan.configurations[configuration_name].batch_size = 2
        plan.configurations[configuration_name].patch_size = (96, 96, 96)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.config_plan.patch_size = (96, 96, 96)
        self.batch_size = 2
        self.num_epochs = 200


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: ViT DINOv3 + Gram Loss Anchoring
# ══════════════════════════════════════════════════════════════════════════════

class BaseDINOv3ViTTrainerWithGram(BaseDINOv3ViTTrainer_NoKoLeo):
    """
    Stage 2: ViT DINOv3 (no KoLeo) + Gram loss for feature anchoring.
    Load Stage 1 teacher checkpoint as gram anchor.

    Usage:
        CUDA_VISIBLE_DEVICES=1 nnssl_train Dataset910_Combined onemmiso \\
            -tr BaseDINOv3ViTTrainerWithGram_BS1_patch128_ep200
        (auto-loads Stage 1 teacher from checkpoint_best_teacher.pth)
    """
    STAGE1_TEACHER_CKPT = (
        '/mnt/all_data/ssl_foundation/data/nnssl_results/Dataset910_Combined/'
        'BaseDINOv3ViTTrainer_NoKoLeo_BS1_patch128_ep200__nnsslPlans__onemmiso/'
        'fold_all/checkpoint_best_teacher.pth'
    )

    def __init__(self, plan, configuration_name, fold, pretrain_json,
                 device=torch.device("cuda")):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.lambda_gram   = 0.01
        self.gram_start_ep = 0

    def initialize(self):
        super().initialize()
        from gram_loss import GramLoss
        self.gram_loss = GramLoss(apply_norm=True, remove_neg=True).to(self.device)

        # Load Stage 1 teacher as gram anchor
        if os.path.exists(self.STAGE1_TEACHER_CKPT):
            self.gram_teacher = copy.deepcopy(self.feature_extractor)
            ckpt = torch.load(self.STAGE1_TEACHER_CKPT,
                              map_location='cpu', weights_only=False)
            state_dict = ckpt.get('teacher', ckpt)
            # Interpolate pos_embed if shape mismatch (e.g. stage2 128->stage3 160)
            if 'encoder.pos_embed' in state_dict:
                ckpt_pe = state_dict['encoder.pos_embed']
                model_pe = self.gram_teacher.encoder.pos_embed
                if ckpt_pe.shape != model_pe.shape:
                    import torch.nn.functional as F
                    # ckpt_pe: [1, N_ckpt, D], model_pe: [1, N_model, D]
                    # Separate cls token and patch tokens
                    cls_token = ckpt_pe[:, :1, :]
                    patch_tokens = ckpt_pe[:, 1:, :]
                    n_model = model_pe.shape[1] - 1
                    # Interpolate patch tokens
                    patch_tokens = patch_tokens.reshape(1, 1, -1, ckpt_pe.shape[-1]).permute(0,3,1,2)
                    patch_tokens = F.interpolate(patch_tokens, size=(1, n_model), mode='nearest')
                    patch_tokens = patch_tokens.permute(0,2,3,1).reshape(1, n_model, ckpt_pe.shape[-1])
                    state_dict['encoder.pos_embed'] = torch.cat([cls_token, patch_tokens], dim=1)
                    print(f'[ViT-Gram] Interpolated pos_embed {ckpt_pe.shape} -> {state_dict["encoder.pos_embed"].shape}')
            self.gram_teacher.load_state_dict(state_dict, strict=False)
            for p in self.gram_teacher.parameters():
                p.requires_grad_(False)
            self.gram_teacher.eval()
            print(f'[ViT-Gram] Stage 1 teacher loaded from {self.STAGE1_TEACHER_CKPT}')
        else:
            # Fallback — use current EMA teacher
            self.gram_teacher = copy.deepcopy(self.teacher_extractor)
            for p in self.gram_teacher.parameters():
                p.requires_grad_(False)
            print('[ViT-Gram] No Stage 1 ckpt found — using current teacher as anchor')

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        if isinstance(data, (list, tuple)): data = data[0]
        data = data.to(self.device, non_blocking=True)

        # Base DINOv3 loss
        loss, l_dino, l_ibot = self._forward_dinov3(data)

        # Gram loss (from Stage 1 teacher → student patches)
        if self.current_epoch >= self.gram_start_ep:
            with torch.no_grad():
                _, g_patches = self.gram_teacher(data)
            _, s_patches = self.feature_extractor(data)
            with autocast(self.device.type, enabled=True):
                l_gram = self.gram_loss(s_patches, g_patches, img_level=True)
            loss = loss + self.lambda_gram * l_gram

        self.optimizer.zero_grad(set_to_none=True)
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(self.feature_extractor.parameters()) +
            list(self.s_cls_head.parameters()) +
            list(self.s_patch_head.parameters()), 1.0)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        m = self._get_ema_momentum()
        ema_update(self.feature_extractor, self.teacher_extractor, m)
        ema_update(self.s_cls_head,   self.t_cls_head,   m)
        ema_update(self.s_patch_head, self.t_patch_head, m)

        self.global_step += 1
        del data
        return {'loss': loss.detach().cpu().numpy()}

    def save_checkpoint(self, filename):
        super().save_checkpoint(filename)
        # Also save gram teacher path for Stage 3
        info_path = filename.replace('.pth', '_gram_info.txt')
        with open(info_path, 'w') as f:
            f.write(f'gram_teacher_src: {self.STAGE1_TEACHER_CKPT}\n')
            f.write(f'epoch: {self.current_epoch}\n')


class BaseDINOv3ViTTrainerWithGram_BS1_patch128_ep200(BaseDINOv3ViTTrainerWithGram):
    """Stage 2: ViT + Gram, BS=1, patch=128³, 200 epochs."""
    def __init__(self, plan, configuration_name, fold, pretrain_json,
                 device=torch.device("cuda")):
        plan.configurations[configuration_name].batch_size = 1
        plan.configurations[configuration_name].patch_size = (128, 128, 128)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.config_plan.patch_size = (128, 128, 128)
        self.batch_size  = 1
        self.num_epochs  = 200


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3: High-Resolution ViT + Gram
# ══════════════════════════════════════════════════════════════════════════════

class BaseDINOv3ViTTrainerHighRes(BaseDINOv3ViTTrainerWithGram):
    """
    Stage 3: ViT DINOv3 at higher resolution (160³) with Gram anchoring.
    Uses Stage 2 teacher as gram anchor.

    Usage (after Stage 2 finishes):
        CUDA_VISIBLE_DEVICES=1 nnssl_train Dataset910_Combined onemmiso \\
            -tr BaseDINOv3ViTTrainerHighRes_160_ep50 --c
    """
    STAGE1_TEACHER_CKPT = (
        '/mnt/all_data/ssl_foundation/data/nnssl_results/Dataset910_Combined/'
        'BaseDINOv3ViTTrainerWithGram_BS1_patch128_ep200__nnsslPlans__onemmiso/'
        'fold_all/checkpoint_best_teacher.pth'
    )

    def __init__(self, plan, configuration_name, fold, pretrain_json,
                 device=torch.device("cuda")):
        plan.configurations[configuration_name].patch_size = (160, 160, 160)
        plan.configurations[configuration_name].batch_size = 1
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.config_plan.patch_size = (160, 160, 160)
        self.batch_size  = 1
        self.initial_lr  = 1e-5   # lower LR for adaptation
        self.lambda_gram = 0.01
        self.gram_start_ep = 0


class BaseDINOv3ViTTrainerHighRes_160_ep50(BaseDINOv3ViTTrainerHighRes):
    """Stage 3: patch=160³, 50 epochs."""
    def __init__(self, plan, configuration_name, fold, pretrain_json,
                 device=torch.device("cuda")):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.num_epochs = 50


class BaseDINOv3ViTTrainerHighRes_192_ep50(BaseDINOv3ViTTrainerHighRes):
    """Stage 3: patch=192³, 50 epochs — maximum resolution."""
    def __init__(self, plan, configuration_name, fold, pretrain_json,
                 device=torch.device("cuda")):
        plan.configurations[configuration_name].patch_size = (192, 192, 192)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.config_plan.patch_size = (192, 192, 192)
        self.num_epochs = 50


# ══════════════════════════════════════════════════════════════════════════════
# Full Pipeline Summary
# ══════════════════════════════════════════════════════════════════════════════
#
# Stage 1 (running now):
#   BaseDINOv3ViTTrainer_NoKoLeo_BS1_patch128_ep200
#   → 200 epochs, patch=128³, DINO+iBOT, no KoLeo
#   → saves checkpoint_best_vit_student.pth  ← LP/EAO after stage 1
#   → saves checkpoint_best_teacher.pth      ← gram anchor for Stage 2
#
# Stage 2 (run after Stage 1):
#   BaseDINOv3ViTTrainerWithGram_BS1_patch128_ep200
#   CUDA_VISIBLE_DEVICES=1 nnssl_train Dataset910_Combined onemmiso \
#       -tr BaseDINOv3ViTTrainerWithGram_BS1_patch128_ep200
#   → 200 epochs, patch=128³, DINO+iBOT+Gram
#   → saves checkpoint_best_vit_student.pth  ← LP/EAO after stage 2
#   → saves checkpoint_best_teacher.pth      ← gram anchor for Stage 3
#
# Stage 3 (run after Stage 2):
#   BaseDINOv3ViTTrainerHighRes_160_ep50
#   CUDA_VISIBLE_DEVICES=1 nnssl_train Dataset910_Combined onemmiso \
#       -tr BaseDINOv3ViTTrainerHighRes_160_ep50
#   → 50 epochs, patch=160³, DINO+iBOT+Gram
#   → saves checkpoint_best_vit_student.pth  ← BEST for LP/EAO
#
# Use for LP extraction:
#   extract_feat_nnssl_vit_lp.py --ckpt checkpoint_best_vit_student.pth
