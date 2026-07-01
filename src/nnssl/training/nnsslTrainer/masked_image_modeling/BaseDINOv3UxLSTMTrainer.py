"""
BaseDINOv3UxLSTMTrainer
========================
nnSSL trainer combining:
  - UxLSTMBot backbone (CNN + xLSTM bottleneck)
  - DINOv3 losses (DINO + iBOT + KoLeo) instead of MAE reconstruction

Usage:
    nnssl_train BaseDINOv3UxLSTMTrainer Dataset910_Combined onemmiso

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

sys.path.insert(0, '/home/aqayyum/Scar_Segmentation_models')

from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainerUxLSTM import BaseMAETrainerUxLSTM, get_uxlstm_bot
from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan
from batchgenerators.utilities.file_and_folder_operations import save_json


# ── DINOv3 Losses ─────────────────────────────────────────────────────────────

class DINOLoss(nn.Module):
    def __init__(self, n_proto=65536, s_temp=0.1, t_temp=0.04, center_mom=0.9):
        super().__init__()
        self.s_temp     = s_temp
        self.t_temp     = t_temp
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
        self.s_temp     = s_temp
        self.t_temp     = t_temp
        self.c_mom      = center_mom
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

    def forward(self, x, eps=1e-3):  # larger eps prevents log(0)
        with torch.autocast("cuda", enabled=False):
            x = F.normalize(x.float(), p=2, dim=-1)
            dots = torch.mm(x, x.t())
            dots.view(-1)[:: (x.shape[0] + 1)].fill_(-1)
            _, idx = torch.max(dots, dim=1)
            dist = self.pdist(x, x[idx]).clamp(min=eps)
            loss = -torch.log(dist).mean()
            # Safety clamp — prevent extreme values
            loss = loss.clamp(max=100.0)
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
        # Normalize weight for stability (weight norm effect)
        w = F.normalize(self.last.weight, dim=1)
        return F.linear(x, w)


@torch.no_grad()
def ema_update(student, teacher, m):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(m).add_(ps.data, alpha=1 - m)


# ── UxLSTM Feature Extractor ──────────────────────────────────────────────────

class UxLSTMFeatureExtractor(nn.Module):
    """
    Wraps UxLSTMBot to extract:
      - CLS token (global avg pool of bottleneck features)
      - Patch tokens (bottleneck feature map flattened)
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        enc = self.backbone.encoder if hasattr(self.backbone, 'encoder') else self.backbone

        # Run stem first (input projection)
        feat = enc.stem(x) if hasattr(enc, 'stem') else x

        # Run all encoder stages
        for stage in enc.stages:
            feat = stage(feat)

        # Bottleneck = deepest encoder output (B, C, D, H, W)
        B, C, *spatial = feat.shape

        # CLS = global average pool → (B, C)
        cls_token = feat.flatten(2).mean(-1)        # (B, C)

        # Patch tokens → (B, N, C)
        patch_tokens = feat.flatten(2).transpose(1, 2)  # (B, N, C)

        return cls_token, patch_tokens


# ── Main Trainer ──────────────────────────────────────────────────────────────

class BaseDINOv3UxLSTMTrainer(BaseMAETrainerUxLSTM):
    """
    UxLSTMBot backbone with DINOv3 losses.
    
    Key differences from BaseMAETrainerUxLSTM:
      Loss:      MAE reconstruction → DINO + iBOT + KoLeo
      Training:  Single encoder → Student/Teacher EMA
      Output:    Reconstructed image → CLS + patch tokens
    """

    def __init__(self, plan, configuration_name, fold, pretrain_json,
                 device=torch.device("cuda")):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        # DINOv3 hyperparams
        self.n_proto      = 65536
        self.s_temp       = 0.1
        self.t_temp_init  = 0.04  # start same as final — prevents early collapse   # warmup start
        self.t_temp_final = 0.04   # warmup end
        self.t_warmup_ep  = 30
        self.center_mom   = 0.9
        self.ema_start    = 0.996
        self.ema_end      = 1.0
        self.lambda_dino  = 1.0
        self.lambda_ibot  = 1.0
        self.lambda_koleo = 0.01  # reduced from 0.1 — KoLeo unstable at high weight
        self.mask_ratio   = 0.5
        self.global_step  = 0
        self.grad_accum_steps = 4  # effective BS=4

    def initialize(self):
        super().initialize()

        # Get bottleneck dim from UxLSTMBot
        # UxLSTMBot features_per_stage[-1] = 320
        embed_dim = 320

        # Wrap backbone for feature extraction
        self.feature_extractor = UxLSTMFeatureExtractor(self.network).to(self.device)

        # Teacher = EMA copy
        self.teacher_extractor = copy.deepcopy(self.feature_extractor)
        for p in self.teacher_extractor.parameters():
            p.requires_grad_(False)

        # Projection heads
        self.s_cls_head   = DINOHead(embed_dim, self.n_proto).to(self.device)
        self.t_cls_head   = copy.deepcopy(self.s_cls_head)
        self.s_patch_head = DINOHead(embed_dim, self.n_proto).to(self.device)
        self.t_patch_head = copy.deepcopy(self.s_patch_head)

        for p in self.t_cls_head.parameters():   p.requires_grad_(False)
        for p in self.t_patch_head.parameters(): p.requires_grad_(False)

        # Losses
        self.dino_loss  = DINOLoss(self.n_proto, self.s_temp,
                                    self.t_temp_final, self.center_mom).to(self.device)
        self.ibot_loss  = iBOTLoss(self.n_proto, self.s_temp,
                                    self.t_temp_final, self.center_mom).to(self.device)
        self.koleo_loss = KoLeoLoss().to(self.device)

        # Add heads to optimizer
        self.optimizer.add_param_group({'params': self.s_cls_head.parameters(),   'lr': self.initial_lr})
        self.optimizer.add_param_group({'params': self.s_patch_head.parameters(), 'lr': self.initial_lr})

    def _get_teacher_temp(self):
        ep = self.current_epoch
        if ep < self.t_warmup_ep:
            t = self.t_temp_init + (self.t_temp_final - self.t_temp_init) * ep / self.t_warmup_ep
        else:
            t = self.t_temp_final
        return t

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

            l_dino  = self.dino_loss(s_cls_p, t_cls_p)
            l_ibot  = self.ibot_loss(s_patch_p, t_patch_p, mask)
            l_koleo = self.koleo_loss(s_cls)
            loss    = (self.lambda_dino  * l_dino  +
                       self.lambda_ibot  * l_ibot  +
                       self.lambda_koleo * l_koleo)
        return loss, l_dino, l_ibot, l_koleo

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        if isinstance(data, (list, tuple)):
            data = data[0]
        data = data.to(self.device, non_blocking=True)

        loss, l_dino, l_ibot, l_koleo = self._forward_dinov3(data)

        self.optimizer.zero_grad(set_to_none=True)
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(self.feature_extractor.parameters()) +
            list(self.s_cls_head.parameters()) +
            list(self.s_patch_head.parameters()), 3.0)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        m = self._get_ema_momentum()
        ema_update(self.feature_extractor, self.teacher_extractor, m)
        ema_update(self.s_cls_head,  self.t_cls_head,  m)
        ema_update(self.s_patch_head, self.t_patch_head, m)

        # --- component loss logging (every 50 steps) ---
        if not hasattr(self, "_loss_log_buffer"):
            self._loss_log_buffer = {"dino": [], "ibot": [], "koleo": []}
        self._loss_log_buffer["dino"].append(l_dino.detach().item())
        self._loss_log_buffer["ibot"].append(l_ibot.detach().item())
        self._loss_log_buffer["koleo"].append(l_koleo.detach().item())
        if self.global_step % 50 == 0:
            d = np.mean(self._loss_log_buffer["dino"])
            i = np.mean(self._loss_log_buffer["ibot"])
            k = np.mean(self._loss_log_buffer["koleo"])
            self.print_to_log_file(
                f"[DINOv3-components] step {self.global_step} | "
                f"dino={d:.4f} ibot={i:.4f} koleo={k:.4f} "
                f"(weighted: dino={self.lambda_dino*d:.4f} "
                f"ibot={self.lambda_ibot*i:.4f} koleo={self.lambda_koleo*k:.4f})"
            )
            self._loss_log_buffer = {"dino": [], "ibot": [], "koleo": []}
        # --- end logging ---

        self.global_step += 1
        del data
        return {"loss": loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        if isinstance(data, (list, tuple)):
            data = data[0]
        data = data.to(self.device, non_blocking=True)

        with torch.no_grad():
            loss, l_dino, l_ibot, l_koleo = self._forward_dinov3(data)

        del data
        return {"loss": loss.detach().cpu().numpy()}

    def run_online_evaluation(self, *args, **kwargs):
        pass

    def finish_online_evaluation(self):
        if self.online_eval_losses:
            mean = np.mean(self.online_eval_losses)
            self.print_to_log_file(
                f"[DINOv3-UxLSTM] Mean loss: {mean:.4f}")
            self.online_eval_losses = []

    def save_checkpoint(self, filename):
        """Override to also save teacher weights."""
        super().save_checkpoint(filename)
        # Save teacher separately alongside checkpoint
        teacher_path = filename.replace(".pth", "_teacher.pth")
        torch.save({
            "teacher":      self.teacher_extractor.state_dict(),
            "t_cls_head":   self.t_cls_head.state_dict(),
            "t_patch_head": self.t_patch_head.state_dict(),
            "epoch":        self.current_epoch,
        }, teacher_path)
        self.print_to_log_file(f"[DINOv3] Teacher saved to {teacher_path}")


# ── Variants ──────────────────────────────────────────────────────────────────

class BaseDINOv3UxLSTMTrainer_BS2(BaseDINOv3UxLSTMTrainer):
    def __init__(self, plan, configuration_name, fold, pretrain_json,
                 device=torch.device("cuda")):
        plan.configurations[configuration_name].batch_size = 2
        super().__init__(plan, configuration_name, fold, pretrain_json, device)


class BaseDINOv3UxLSTMTrainer_BS2_ep200(BaseDINOv3UxLSTMTrainer):
    def __init__(self, plan, configuration_name, fold, pretrain_json,
                 device=torch.device("cuda")):
        plan.configurations[configuration_name].batch_size = 2
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.num_epochs = 200


class BaseDINOv3UxLSTMTrainer_BS4_ep200(BaseDINOv3UxLSTMTrainer):
    def __init__(self, plan, configuration_name, fold, pretrain_json,
                 device=torch.device("cuda")):
        plan.configurations[configuration_name].batch_size = 4
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.num_epochs = 200


class BaseDINOv3UxLSTMTrainer_BS1_patch128(BaseDINOv3UxLSTMTrainer):
    """BS=1, patch 128^3, lower LR for stability."""
    def __init__(self, plan, configuration_name, fold, pretrain_json,
                 device=torch.device("cuda")):
        plan.configurations[configuration_name].batch_size = 1
        plan.configurations[configuration_name].patch_size = (128, 128, 128)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.config_plan.patch_size = (128, 128, 128)
        self.batch_size = 1
        self.initial_lr  = 1e-3   # reduced from 0.01 → 0.001
        self.weight_decay = 0.04


# Key change: Removed KoLeoLoss entirely and all related usage

class BaseDINOv3UxLSTMTrainer_NoKoLeo(BaseMAETrainerUxLSTM):
    def __init__(self, plan, configuration_name, fold, pretrain_json,
                 device=torch.device("cuda")):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)

        self.n_proto      = 65536
        self.s_temp       = 0.1
        self.t_temp_init  = 0.04
        self.t_temp_final = 0.04
        self.t_warmup_ep  = 30
        self.center_mom   = 0.9
        self.ema_start    = 0.996
        self.ema_end      = 1.0

        self.lambda_dino  = 1.0
        self.lambda_ibot  = 1.0

        self.mask_ratio   = 0.5
        self.global_step  = 0
        self.grad_accum_steps = 4

    def initialize(self):
        super().initialize()

        embed_dim = 320

        self.feature_extractor = UxLSTMFeatureExtractor(self.network).to(self.device)

        self.teacher_extractor = copy.deepcopy(self.feature_extractor)
        for p in self.teacher_extractor.parameters():
            p.requires_grad_(False)

        self.s_cls_head   = DINOHead(embed_dim, self.n_proto).to(self.device)
        self.t_cls_head   = copy.deepcopy(self.s_cls_head)
        self.s_patch_head = DINOHead(embed_dim, self.n_proto).to(self.device)
        self.t_patch_head = copy.deepcopy(self.s_patch_head)

        for p in self.t_cls_head.parameters():   p.requires_grad_(False)
        for p in self.t_patch_head.parameters(): p.requires_grad_(False)

        self.dino_loss  = DINOLoss(self.n_proto, self.s_temp,
                                  self.t_temp_final, self.center_mom).to(self.device)
        self.ibot_loss  = iBOTLoss(self.n_proto, self.s_temp,
                                  self.t_temp_final, self.center_mom).to(self.device)

        self.optimizer.add_param_group({'params': self.s_cls_head.parameters(),   'lr': self.initial_lr})
        self.optimizer.add_param_group({'params': self.s_patch_head.parameters(), 'lr': self.initial_lr})

    def _get_teacher_temp(self):
        ep = self.current_epoch
        if ep < self.t_warmup_ep:
            t = self.t_temp_init + (self.t_temp_final - self.t_temp_init) * ep / self.t_warmup_ep
        else:
            t = self.t_temp_final
        return t

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

            l_dino  = self.dino_loss(s_cls_p, t_cls_p)
            l_ibot  = self.ibot_loss(s_patch_p, t_patch_p, mask)

            loss = (self.lambda_dino * l_dino +
                    self.lambda_ibot * l_ibot)

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
            list(self.s_patch_head.parameters()), 3.0)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        m = self._get_ema_momentum()
        ema_update(self.feature_extractor, self.teacher_extractor, m)
        ema_update(self.s_cls_head,  self.t_cls_head,  m)
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
            loss, l_dino, l_ibot = self._forward_dinov3(data)

        del data
        return {"loss": loss.detach().cpu().numpy()}
















# ── Stage 2: Gram Anchoring Base ────────────────────────────────────────────

class BaseDINOv3UxLSTMTrainerWithGram(BaseDINOv3UxLSTMTrainer):
    """Stage 2: DINOv3 + Gram loss for feature distribution anchoring."""
    def __init__(self, plan, configuration_name, fold, pretrain_json,
                 device=torch.device("cuda"), gram_ckpt=None):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.gram_ckpt   = gram_ckpt
        self.lambda_gram = 1.0
        self.gram_start_ep = 0

    def initialize(self):
        super().initialize()
        from gram_loss import GramLoss
        self.gram_loss = GramLoss(apply_norm=True, remove_neg=True).to(self.device)

        if self.gram_ckpt and os.path.exists(self.gram_ckpt):
            self.gram_teacher = copy.deepcopy(self.feature_extractor)
            ckpt = torch.load(self.gram_ckpt, map_location="cpu", weights_only=False)
            self.gram_teacher.load_state_dict(ckpt.get("teacher", ckpt))
            for p in self.gram_teacher.parameters():
                p.requires_grad_(False)
            print(f"[DINOv3] Gram teacher loaded from {self.gram_ckpt}")
        else:
            self.gram_teacher = copy.deepcopy(self.teacher_extractor)
            print("[DINOv3] No gram ckpt — using current teacher as gram anchor")

    def run_iteration(self, batch, do_backprop=True, run_online_evaluation=False):
        loss_val = super().run_iteration(batch, do_backprop=do_backprop,
                                          run_online_evaluation=run_online_evaluation)
        if self.current_epoch >= self.gram_start_ep:
            data = batch["data"]
            if isinstance(data, (list, tuple)):
                data = data[0]
            data = data.to(self.device, non_blocking=True)
            with torch.no_grad():
                _, g_patches = self.gram_teacher(data)
            _, s_patches = self.feature_extractor(data)
            with autocast(self.device.type, enabled=True):
                l_gram = self.gram_loss(s_patches, g_patches, img_level=True)
            if do_backprop:
                self.optimizer.zero_grad(set_to_none=True)
                self.grad_scaler.scale(l_gram * self.lambda_gram).backward()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
        return loss_val


# ── Stage 2: Gram Anchoring Variants ─────────────────────────────────────────

class BaseDINOv3UxLSTMTrainerWithGram_BS1_patch128(BaseDINOv3UxLSTMTrainerWithGram):
    """
    Stage 2: DINOv3 + Gram loss, BS=1, patch 128^3.
    Requires gram_ckpt from Stage 1 teacher checkpoint.
    Usage:
        CUDA_VISIBLE_DEVICES=0 nnssl_train Dataset910_Combined onemmiso \
            -tr BaseDINOv3UxLSTMTrainerWithGram_BS1_patch128 \
            --gram_ckpt /path/to/stage1/teacher.pth
    """
    def __init__(self, plan, configuration_name, fold, pretrain_json,
                 device=torch.device("cuda"), gram_ckpt=None):
        plan.configurations[configuration_name].batch_size = 1
        plan.configurations[configuration_name].patch_size = (128, 128, 128)
        super().__init__(plan, configuration_name, fold, pretrain_json,
                         device, gram_ckpt=gram_ckpt)
        self.config_plan.patch_size = (128, 128, 128)
        self.batch_size = 1
        self.gram_start_ep = 0   # start gram loss from epoch 0


class BaseDINOv3UxLSTMTrainerWithGram_BS1_patch128_ep200(BaseDINOv3UxLSTMTrainerWithGram_BS1_patch128):
    """Stage 2, 200 epochs."""
    def __init__(self, plan, configuration_name, fold, pretrain_json,
                 device=torch.device("cuda"), gram_ckpt=None):
        super().__init__(plan, configuration_name, fold, pretrain_json,
                         device, gram_ckpt=gram_ckpt)
        self.num_epochs = 200
        
## Stage 1 teacher checkpoint location
#GRAM_CKPT="/mnt/all_data/ssl_foundation/data/nnssl_results/Dataset910_Combined/\
#BaseDINOv3UxLSTMTrainer_BS1_patch128__nnsslPlans__onemmiso/fold_all/\
#checkpoint_best_teacher.pth"
#
## Run Stage 2
#CUDA_VISIBLE_DEVICES=0 nnssl_train Dataset910_Combined onemmiso \
#    -tr BaseDINOv3UxLSTMTrainerWithGram_BS1_patch128_ep200 \
#    2>&1 | tee /mnt/all_data/ssl_foundation/data/nnssl_results/dinov3_xlstm_stage2.log


# ── Stage 3: High-Resolution Adaptation ───────────────────────────────────────

class BaseDINOv3UxLSTMTrainerHighRes(BaseDINOv3UxLSTMTrainerWithGram):
    """
    Stage 3: High-resolution adaptation for CT data.
    
    Usage (after Stage 2 completes):
        CUDA_VISIBLE_DEVICES=0 nnssl_train Dataset910_Combined onemmiso \\
            -tr BaseDINOv3UxLSTMTrainerHighRes_160_ep50 \\
            --c

    Requires Stage 2 checkpoint in output folder.
    Uses Gram loss to anchor dense features from Stage 2 teacher.

    Resolution progression:
        Stage 1: 128³  (current)
        Stage 2: 128³  + Gram anchoring
        Stage 3: 160³  + Gram anchoring  ← this trainer
    """
    def __init__(self, plan, configuration_name, fold, pretrain_json,
                 device=torch.device("cuda"), gram_ckpt=None):
        # High-res patch size
        plan.configurations[configuration_name].patch_size = (160, 160, 160)
        plan.configurations[configuration_name].batch_size = 1
        super().__init__(plan, configuration_name, fold, pretrain_json,
                         device, gram_ckpt=gram_ckpt)
        self.config_plan.patch_size = (160, 160, 160)
        self.batch_size = 1
        # Use smaller LR for adaptation
        self.initial_lr  = 1e-5
        self.weight_decay = 0.04
        # Gram loss active from start
        self.gram_start_ep = 0
        self.lambda_gram   = 1.0


class BaseDINOv3UxLSTMTrainerHighRes_160_ep50(BaseDINOv3UxLSTMTrainerHighRes):
    """Stage 3, patch=160³, 50 epochs."""
    def __init__(self, plan, configuration_name, fold, pretrain_json,
                 device=torch.device("cuda"), gram_ckpt=None):
        super().__init__(plan, configuration_name, fold, pretrain_json,
                         device, gram_ckpt=gram_ckpt)
        self.num_epochs = 50


class BaseDINOv3UxLSTMTrainerHighRes_192_ep50(BaseDINOv3UxLSTMTrainerHighRes):
    """Stage 3, patch=192³, 50 epochs — maximum resolution."""
    def __init__(self, plan, configuration_name, fold, pretrain_json,
                 device=torch.device("cuda"), gram_ckpt=None):
        plan.configurations[configuration_name].patch_size = (192, 192, 192)
        super().__init__(plan, configuration_name, fold, pretrain_json,
                         device, gram_ckpt=gram_ckpt)
        self.config_plan.patch_size = (192, 192, 192)
        self.num_epochs = 50
        
        
        
        
## After Stage 2 finishes, run Stage 3 with Stage 2 teacher as gram anchor
#GRAM_CKPT="/mnt/all_data/ssl_foundation/data/nnssl_results/Dataset910_Combined/\
#BaseDINOv3UxLSTMTrainerWithGram_BS1_patch128_ep200__nnsslPlans__onemmiso/fold_all/\
#checkpoint_best_teacher.pth"
#
#CUDA_VISIBLE_DEVICES=0 nnssl_train Dataset910_Combined onemmiso \
#    -tr BaseDINOv3UxLSTMTrainerHighRes_160_ep50 \
#    2>&1 | tee dinov3_xlstm_stage3.log


#Full pipeline when you're ready:
#Stage 1: BaseDINOv3UxLSTMTrainer_BS1_patch128        ← running now (200 ep)
#Stage 2: BaseDINOv3UxLSTMTrainerWithGram_BS1_patch128_ep200  (100 ep)
#Stage 3: BaseDINOv3UxLSTMTrainerHighRes_160_ep50      (50 ep)

############################# complete desciption of method ############################

#Stage 1:  128³  DINO + iBOT + KoLeo
#Stage 2:  128³  DINO + iBOT + KoLeo + Gram  ← same resolution, add Gram
#Stage 3:  160³  DINO + iBOT + KoLeo + Gram  ← higher resolution, same losses
#The purpose of each stage:
#Stage 1: Learn representations from scratch
#         → encoder learns what CT anatomy looks like
#         → free exploration, no constraints
#
#Stage 2: Stabilize before resolution change  
#         → Gram anchors the feature structure
#         → prevents drift when you increase resolution
#         → same resolution so model just stabilizes
#
#Stage 3: Adapt to higher resolution
#         → model sees larger context (160³ vs 128³)
#         → Gram loss keeps dense features from drifting
#         → better for downstream tasks needing fine detail
#For CT segmentation specifically:
#128³ → sees ~20cm × 20cm × 20cm region
#160³ → sees ~25cm × 25cm × 25cm region
#192³ → sees ~30cm × 30cm × 30cm region
#Larger crops = model understands more anatomical context = better segmentation of large structures (liver, lung, etc.)
#For small structures (lesions, vessels) — 128³ is already sufficient.

#What makes it publishable
#The key comparison for the paper:
#Baseline 1: nnSSL + UxLSTM + MAE loss     (existing)
#Baseline 2: nnSSL + ResNet + DINOv3 loss  (ablation)
#Proposed:   nnSSL + UxLSTM + DINOv3 loss  (yours)
#
#Evaluate on: AMOS, BTCV, MSD segmentation benchmarks
#Metric: Dice score with 1%, 10%, 100% labelled data
#If your model shows +2-3% Dice over MAE baseline → strong MICCAI/MedIA paper.

#################################### Model usage ####################################################


#Never use teacher for downstream — teacher has no decoder, only encoder + projection heads which are discarded after pretraining.
#Full pipeline:
#Stage 1:  BaseDINOv3UxLSTMTrainer_BS1_patch128
#          → saves checkpoint_best.pth          ← Stage 1 student
#          → saves checkpoint_best_teacher.pth  ← gram anchor for Stage 2
#
#Stage 2:  BaseDINOv3UxLSTMTrainerWithGram_BS1_patch128_ep200
#          gram_ckpt = Stage1/checkpoint_best_teacher.pth
#          → saves checkpoint_best.pth          ← USE THIS downstream
#          → saves checkpoint_best_teacher.pth  ← gram anchor for Stage 3
#
#Stage 3:  BaseDINOv3UxLSTMTrainerHighRes_160_ep50
#          gram_ckpt = Stage2/checkpoint_best_teacher.pth
#          → saves checkpoint_best.pth          ← BEST for downstream
#Rule:
#Always use the LATEST stage student checkpoint_best.pth for downstream
#Each stage improves on the previous







class BaseDINOv3UxLSTMTrainer_Stable(BaseMAETrainerUxLSTM):
    """
    Stable DINOv3 UxLSTM trainer without KoLeo.
    Fixes:
      1. No KoLeo loss (main NaN source)
      2. Center clipping to prevent divergence
      3. Gradient accumulation (effective BS=4)
      4. NaN guard with EMA skip
      5. Lower LR (1e-3)
      6. Center reset if diverges
    """
    def __init__(self, plan, configuration_name, fold, pretrain_json,
                 device=torch.device("cuda")):
        plan.configurations[configuration_name].batch_size = 1
        plan.configurations[configuration_name].patch_size = (128, 128, 128)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.config_plan.patch_size = (128, 128, 128)
        self.batch_size = 1

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
        self.initial_lr   = 1e-4  # lowered from 1e-3 - was causing inf grads / collapse

    def build_architecture_and_adaptation_plan(self, *args, **kwargs):
        network, adapt_plan = super().build_architecture_and_adaptation_plan(*args, **kwargs)
        if hasattr(adapt_plan, "key_to_xlstm"):
            adapt_plan.key_to_xlstm = None
        return network, adapt_plan

    def initialize(self):
        super().initialize()
        embed_dim = 320

        self.feature_extractor = UxLSTMFeatureExtractor(self.network).to(self.device)
        self.teacher_extractor = copy.deepcopy(self.feature_extractor)
        for p in self.teacher_extractor.parameters():
            p.requires_grad_(False)

        self.s_cls_head   = DINOHead(embed_dim, self.n_proto).to(self.device)
        self.t_cls_head   = copy.deepcopy(self.s_cls_head)
        self.s_patch_head = DINOHead(embed_dim, self.n_proto).to(self.device)
        self.t_patch_head = copy.deepcopy(self.s_patch_head)
        for p in self.t_cls_head.parameters():   p.requires_grad_(False)
        for p in self.t_patch_head.parameters(): p.requires_grad_(False)

        self.dino_loss = DINOLoss(self.n_proto, self.s_temp,
                                   self.t_temp_final, self.center_mom).to(self.device)
        self.ibot_loss = iBOTLoss(self.n_proto, self.s_temp,
                                   self.t_temp_final, self.center_mom).to(self.device)

        self.optimizer.add_param_group({"params": list(self.s_cls_head.parameters()) +
                                                   list(self.s_patch_head.parameters()),
                                        "lr": self.initial_lr})

    def _get_teacher_temp(self):
        ep = self.current_epoch
        if ep < self.t_warmup_ep:
            return self.t_temp_init + (self.t_temp_final - self.t_temp_init) * ep / self.t_warmup_ep
        return self.t_temp_final

    def _get_ema_momentum(self):
        total = self.num_epochs * 250
        return self.ema_end - (self.ema_end - self.ema_start) *                (np.cos(np.pi * self.global_step / total) + 1) / 2

    def _forward_dinov3(self, data):
        t_temp = self._get_teacher_temp()
        self.dino_loss.t_temp = t_temp
        self.ibot_loss.t_temp = t_temp

        # Clip centers to prevent divergence
        with torch.no_grad():
            self.dino_loss.center.clamp_(-10, 10)
            self.ibot_loss.center.clamp_(-10, 10)

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

            # Safety clamp on individual losses
            l_dino = l_dino.clamp(max=50.0)
            l_ibot = l_ibot.clamp(max=50.0)

            loss = self.lambda_dino * l_dino + self.lambda_ibot * l_ibot

        return loss, l_dino, l_ibot

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        if isinstance(data, (list, tuple)): data = data[0]
        data = data.to(self.device, non_blocking=True)

        loss, l_dino, l_ibot = self._forward_dinov3(data)

        # NaN guard — skip step AND EMA
        if not torch.isfinite(loss):
            self.print_to_log_file(f"[Stable] NaN at step {self.global_step} — skipping")
            # Reset centers if NaN
            self.dino_loss.center.zero_()
            self.ibot_loss.center.zero_()
            del data
            self.global_step += 1
            return {"loss": getattr(self, "prev_loss", 0.0)}

        # Gradient accumulation
        loss_scaled = loss / self.grad_accum_steps
        self.grad_scaler.scale(loss_scaled).backward()

        skip_ema = False
        if (self.global_step + 1) % self.grad_accum_steps == 0:
            self.grad_scaler.unscale_(self.optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(
                list(self.feature_extractor.parameters()) +
                list(self.s_cls_head.parameters()) +
                list(self.s_patch_head.parameters()), 1.0)
            if torch.isfinite(total_norm):
                self.grad_scaler.step(self.optimizer)
            else:
                self.print_to_log_file(f"[Stable] Inf grad at step {self.global_step} - skipping optimizer AND EMA")
                skip_ema = True
            self.grad_scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        # EMA update - skipped if this accumulation window had an inf grad
        if not skip_ema:
            m = self._get_ema_momentum()
            ema_update(self.feature_extractor, self.teacher_extractor, m)
            ema_update(self.s_cls_head,  self.t_cls_head,  m)
            ema_update(self.s_patch_head, self.t_patch_head, m)

        # --- component loss logging (every 50 steps) ---
        if not hasattr(self, "_loss_log_buffer"):
            self._loss_log_buffer = {"dino": [], "ibot": []}
        self._loss_log_buffer["dino"].append(l_dino.detach().item())
        self._loss_log_buffer["ibot"].append(l_ibot.detach().item())
        if self.global_step % 50 == 0:
            d = np.mean(self._loss_log_buffer["dino"])
            i = np.mean(self._loss_log_buffer["ibot"])
            self.print_to_log_file(
                f"[Stable-components] step {self.global_step} | "
                f"dino={d:.4f} ibot={i:.4f} "
                f"(weighted: dino={self.lambda_dino*d:.4f} ibot={self.lambda_ibot*i:.4f})"
            )
            self._loss_log_buffer = {"dino": [], "ibot": []}
        # --- end logging ---

        loss_val = loss.item()
        self.prev_loss = loss_val
        self.global_step += 1
        del data
        return {"loss": loss_val}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        if isinstance(data, (list, tuple)): data = data[0]
        data = data.to(self.device, non_blocking=True)
        with torch.no_grad():
            loss, l_dino, l_ibot = self._forward_dinov3(data)
        del data
        if not torch.isfinite(loss):
            return {"loss": getattr(self, "prev_loss", 0.0)}
        return {"loss": loss.item()}

    def save_checkpoint(self, filename):
        super().save_checkpoint(filename)
        teacher_path = filename.replace(".pth", "_teacher.pth")
        torch.save({
            "teacher":      self.teacher_extractor.state_dict(),
            "t_cls_head":   self.t_cls_head.state_dict(),
            "t_patch_head": self.t_patch_head.state_dict(),
            "epoch":        self.current_epoch,
            "dino_center":  self.dino_loss.center,
            "ibot_center":  self.ibot_loss.center,
        }, teacher_path)


# ── Smoke-test variants (25 epochs, hardcoded gram_ckpt) ───────────────────
# Quick functional checks for the Stage 1 -> Stage 2 -> Stage 3 pipeline
# before committing to full-length runs. Not for production training.

_STAGE1_STABLE_TEACHER_CKPT = (
    "/mnt/all_data/Abdul/ssl_foundation/data/nnssl_results/Dataset701_UKBB_MRI/"
    "BaseDINOv3UxLSTMTrainer_Stable__nnsslPlans__onemmiso/fold_all/"
    "checkpoint_final_teacher.pth"
)


class BaseDINOv3UxLSTMTrainerWithGram_Smoke25(BaseDINOv3UxLSTMTrainerWithGram_BS1_patch128):
    """
    Stage 2 smoke test: 25 epochs, gram_ckpt hardcoded to Stage 1 (_Stable)
    final teacher checkpoint. For pipeline verification only.
    """
    def __init__(self, plan, configuration_name, fold, pretrain_json,
                 device=torch.device("cuda")):
        super().__init__(plan, configuration_name, fold, pretrain_json,
                          device, gram_ckpt=_STAGE1_STABLE_TEACHER_CKPT)
        self.num_epochs = 25


class BaseDINOv3UxLSTMTrainerHighRes_Smoke25(BaseDINOv3UxLSTMTrainerHighRes):
    """
    Stage 3 smoke test: 25 epochs, gram_ckpt hardcoded to Stage 2 Smoke25's
    final teacher checkpoint. For pipeline verification only.
    """
    def __init__(self, plan, configuration_name, fold, pretrain_json,
                 device=torch.device("cuda")):
        stage2_ckpt = (
            "/mnt/all_data/Abdul/ssl_foundation/data/nnssl_results/Dataset701_UKBB_MRI/"
            "BaseDINOv3UxLSTMTrainerWithGram_Smoke25__nnsslPlans__onemmiso/fold_all/"
            "checkpoint_final_teacher.pth"
        )
        super().__init__(plan, configuration_name, fold, pretrain_json,
                          device, gram_ckpt=stage2_ckpt)
        self.num_epochs = 25


# ── Real Stage 2/3 runs with checkpoint paths wired in ──────────────────────
# Full, unmodified production trainers (BaseDINOv3UxLSTMTrainerWithGram_BS1_patch128_ep200
# and BaseDINOv3UxLSTMTrainerHighRes_160_ep50) with gram_ckpt set to the actual
# checkpoint from the prior stage, since the CLI has no --gram_ckpt flag.

class BaseDINOv3UxLSTMTrainerWithGram_701(BaseDINOv3UxLSTMTrainerWithGram_BS1_patch128_ep200):
    """Stage 2 for Dataset701_UKBB_MRI, gram_ckpt = Stage 1 (_Stable) final teacher."""
    def __init__(self, plan, configuration_name, fold, pretrain_json,
                 device=torch.device("cuda")):
        super().__init__(plan, configuration_name, fold, pretrain_json,
                          device, gram_ckpt=(
            "/mnt/all_data/Abdul/ssl_foundation/data/nnssl_results/Dataset701_UKBB_MRI/"
            "BaseDINOv3UxLSTMTrainer_Stable__nnsslPlans__onemmiso/fold_all/"
            "checkpoint_final_teacher.pth"
        ))


class BaseDINOv3UxLSTMTrainerHighRes_701(BaseDINOv3UxLSTMTrainerHighRes_160_ep50):
    """Stage 3 for Dataset701_UKBB_MRI, gram_ckpt = Stage 2 (701) teacher checkpoint."""
    def __init__(self, plan, configuration_name, fold, pretrain_json,
                 device=torch.device("cuda")):
        super().__init__(plan, configuration_name, fold, pretrain_json,
                          device, gram_ckpt=(
            "/mnt/all_data/Abdul/ssl_foundation/data/nnssl_results/Dataset701_UKBB_MRI/"
            "BaseDINOv3UxLSTMTrainerWithGram_701__nnsslPlans__onemmiso/fold_all/"
            "checkpoint_latest_teacher.pth"
        ))
