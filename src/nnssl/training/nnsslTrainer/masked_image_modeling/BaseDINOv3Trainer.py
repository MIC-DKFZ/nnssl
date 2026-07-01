"""
BaseDINOv3Trainer
=================
nnSSL trainer using DINOv3 losses instead of MAE reconstruction:
  - DINO loss  (CLS token self-distillation)
  - iBOT loss  (patch token masked image modeling)
  - KoLeo loss (entropy regularization — prevents collapse)

Replaces MAEMSELoss with DINOv3 objectives.
Architecture: Student/Teacher ViT with EMA update.
"""

import os, sys, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import autocast

sys.path.insert(0, '/mnt/all_data/Abdul/Heart_vjepa2/Heart_dinov3')

from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import BaseMAETrainer
from nnssl.experiment_planning.experiment_planners.plan import Plan


# ── DINOv3 Losses ────────────────────────────────────────────────────────────

class DINOLoss(nn.Module):
    """DINO CLS token self-distillation loss."""
    def __init__(self, n_prototypes=65536, student_temp=0.1,
                 teacher_temp=0.04, center_momentum=0.9):
        super().__init__()
        self.student_temp   = student_temp
        self.t_temp         = teacher_temp
        self.center_mom     = center_momentum
        self.register_buffer("center", torch.zeros(1, n_prototypes))

    @torch.no_grad()
    def update_center(self, teacher_out):
        self.center = (self.center * self.center_mom
                      + teacher_out.mean(0, keepdim=True) * (1 - self.center_mom))

    def forward(self, student_out, teacher_out):
        s = F.softmax(student_out / self.student_temp, dim=-1)
        t = F.softmax((teacher_out - self.center) / self.t_temp, dim=-1).detach()
        loss = -(t * torch.log(s + 1e-8)).sum(dim=-1).mean()
        self.update_center(teacher_out)
        return loss


class iBOTLoss(nn.Module):
    """iBOT patch token masked image modeling loss."""
    def __init__(self, n_prototypes=65536, student_temp=0.1,
                 teacher_temp=0.04, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.t_temp       = teacher_temp
        self.c_mom        = center_momentum
        self.register_buffer("center", torch.zeros(1, 1, n_prototypes))

    @torch.no_grad()
    def update_center(self, teacher_patches):
        self.center = (self.center * self.c_mom
                      + teacher_patches.mean((0,1), keepdim=True) * (1 - self.c_mom))

    def forward(self, student_patches, teacher_patches, mask):
        s = F.softmax(student_patches / self.student_temp, dim=-1)
        t = F.softmax((teacher_patches - self.center) / self.t_temp, dim=-1).detach()
        loss = -(t * torch.log(s + 1e-8)).sum(dim=-1)
        # Only compute loss on masked patches
        if mask is not None:
            loss = (loss * mask.float()).sum() / (mask.float().sum() + 1e-8)
        else:
            loss = loss.mean()
        self.update_center(teacher_patches)
        return loss


class KoLeoLoss(nn.Module):
    """KoLeo entropy regularization — prevents feature collapse."""
    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def forward(self, student_output, eps=1e-8):
        with torch.autocast("cuda", enabled=False):
            x = F.normalize(student_output.float(), p=2, dim=-1)
            dots = torch.mm(x, x.t())
            n = x.shape[0]
            dots.view(-1)[:: (n + 1)].fill_(-1)  # zero diagonal
            _, indices = torch.max(dots, dim=1)
            distances = self.pdist(x, x[indices])
            loss = -torch.log(distances + eps).mean()
        return loss


# ── EMA Update ───────────────────────────────────────────────────────────────

@torch.no_grad()
def ema_update(student, teacher, momentum):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data = pt.data * momentum + ps.data * (1 - momentum)


def ema_momentum(step, total, start=0.996, end=1.0):
    return end - (end - start) * (np.cos(np.pi * step / total) + 1) / 2


# ── Projection Head ──────────────────────────────────────────────────────────

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=65536, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.last_layer(x)


# ── Main Trainer ─────────────────────────────────────────────────────────────

class BaseDINOv3Trainer(BaseMAETrainer):
    """
    nnSSL trainer with DINOv3 losses.
    Inherits data loading and nnUNet planning from BaseMAETrainer.
    Replaces MAE loss with DINO + iBOT + KoLeo.
    """

    def __init__(self, plan: Plan, configuration_name: str, fold: int,
                 pretrain_json: dict, device=torch.device("cuda")):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)

        # DINOv3 config
        self.n_prototypes    = 65536
        self.student_temp    = 0.1
        self.teacher_temp    = 0.04
        self.teacher_warmup  = 30       # epochs for teacher temp warmup
        self.center_momentum = 0.9
        self.ema_start       = 0.996
        self.ema_end         = 1.0
        self.lambda_dino     = 1.0
        self.lambda_ibot     = 1.0
        self.lambda_koleo    = 0.1
        self.mask_ratio      = 0.5     # fraction of patches to mask for iBOT

    def initialize(self):
        super().initialize()

        embed_dim = self.network.encoder.embed_dim \
            if hasattr(self.network, 'encoder') else 768

        # Student = self.network (already initialized by parent)
        # Teacher = EMA copy
        self.teacher = copy.deepcopy(self.network)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # Projection heads
        self.s_cls_head   = DINOHead(embed_dim, self.n_prototypes).to(self.device)
        self.t_cls_head   = copy.deepcopy(self.s_cls_head)
        self.s_patch_head = DINOHead(embed_dim, self.n_prototypes).to(self.device)
        self.t_patch_head = copy.deepcopy(self.s_patch_head)

        for p in self.t_cls_head.parameters():   p.requires_grad_(False)
        for p in self.t_patch_head.parameters(): p.requires_grad_(False)

        # Losses
        self.dino_loss  = DINOLoss(self.n_prototypes, self.student_temp,
                                    self.teacher_temp, self.center_momentum).to(self.device)
        self.ibot_loss  = iBOTLoss(self.n_prototypes, self.student_temp,
                                    self.teacher_temp, self.center_momentum).to(self.device)
        self.koleo_loss = KoLeoLoss().to(self.device)

        # Add heads to optimizer
        self.optimizer.add_param_group({'params': self.s_cls_head.parameters()})
        self.optimizer.add_param_group({'params': self.s_patch_head.parameters()})

        self.global_step = 0

    def _get_teacher_temp(self, epoch):
        """Warmup teacher temperature."""
        if epoch < self.teacher_warmup:
            return self.teacher_temp * epoch / self.teacher_warmup
        return self.teacher_temp

    def _make_patch_mask(self, n_patches, batch_size):
        """Random mask for iBOT — mask mask_ratio of patches."""
        mask = torch.rand(batch_size, n_patches) < self.mask_ratio
        return mask.to(self.device)

    def run_iteration(self, batch, do_backprop=True, run_online_evaluation=False):
        data = batch['data']
        if isinstance(data, (list, tuple)):
            data = data[0]
        data = data.to(self.device, non_blocking=True)

        # EMA momentum
        total_steps = self.num_epochs * 250  # approx
        momentum = ema_momentum(self.global_step, total_steps, self.ema_start, self.ema_end)

        # Teacher temp warmup
        self.dino_loss.t_temp  = self._get_teacher_temp(self.current_epoch)
        self.ibot_loss.t_temp  = self._get_teacher_temp(self.current_epoch)

        # ── Forward passes ───────────────────────────────────────────────────
        with autocast(self.device.type, enabled=True):

            # Student forward — get CLS + patch tokens
            s_out = self._get_tokens(self.network, data)        # (B, N+1, D)
            s_cls    = s_out[:, 0]                              # (B, D)
            s_patches = s_out[:, 1:]                            # (B, N, D)

            # Teacher forward (no grad)
            with torch.no_grad():
                t_out = self._get_tokens(self.teacher, data)
                t_cls    = t_out[:, 0]
                t_patches = t_out[:, 1:]

            # Projection heads
            s_cls_proj   = self.s_cls_head(s_cls)
            t_cls_proj   = self.t_cls_head(t_cls)
            s_patch_proj = self.s_patch_head(s_patches)
            t_patch_proj = self.t_patch_head(t_patches)

            # Patch mask for iBOT
            B, N, _ = s_patches.shape
            mask = self._make_patch_mask(N, B)

            # ── Losses ───────────────────────────────────────────────────────
            l_dino  = self.dino_loss(s_cls_proj, t_cls_proj)
            l_ibot  = self.ibot_loss(s_patch_proj, t_patch_proj, mask)
            l_koleo = self.koleo_loss(s_cls)
            loss    = (self.lambda_dino  * l_dino +
                       self.lambda_ibot  * l_ibot +
                       self.lambda_koleo * l_koleo)

        if do_backprop:
            self.optimizer.zero_grad(set_to_none=True)
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.network.parameters()) +
                list(self.s_cls_head.parameters()) +
                list(self.s_patch_head.parameters()), 3.0)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            # EMA teacher update
            ema_update(self.network, self.teacher, momentum)
            ema_update(self.s_cls_head, self.t_cls_head, momentum)
            ema_update(self.s_patch_head, self.t_patch_head, momentum)

        self.global_step += 1

        if run_online_evaluation:
            self.run_online_evaluation(loss, l_dino, l_ibot, l_koleo)

        return loss.detach().cpu().numpy()

    def _get_tokens(self, model, x):
        """Extract CLS + patch tokens from encoder."""
        enc = model.encoder if hasattr(model, 'encoder') else model
        return enc(x)  # expects (B, C, D, H, W) → (B, N+1, D)

    def run_online_evaluation(self, loss, l_dino, l_ibot, l_koleo):
        self.online_eval_losses.append(loss.detach().cpu().numpy())
        print(f"  Loss={loss.item():.4f} | DINO={l_dino.item():.4f} "
              f"| iBOT={l_ibot.item():.4f} | KoLeo={l_koleo.item():.4f}")

    def finish_online_evaluation(self):
        if self.online_eval_losses:
            mean = np.mean(self.online_eval_losses)
            self.print_to_log_file(f"Mean DINOv3 loss: {mean:.4f}")
            self.online_eval_losses = []


class BaseDINOv3TrainerWithGram(BaseDINOv3Trainer):
    """
    Stage 2: DINOv3 + Gram anchoring.
    Use after Stage 1 to prevent dense feature degradation.
    Requires a frozen gram teacher checkpoint.
    """
    def __init__(self, plan, configuration_name, fold, pretrain_json,
                 device=torch.device("cuda"), gram_ckpt=None):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.gram_ckpt     = gram_ckpt
        self.lambda_gram   = 1.0
        self.gram_start_ep = 0  # epoch to start gram loss

    def initialize(self):
        super().initialize()
        from gram_loss import GramLoss
        self.gram_loss = GramLoss(apply_norm=True, remove_neg=True).to(self.device)

        # Load frozen gram teacher
        if self.gram_ckpt and os.path.exists(self.gram_ckpt):
            self.gram_teacher = copy.deepcopy(self.network)
            ckpt = torch.load(self.gram_ckpt, map_location='cpu', weights_only=False)
            self.gram_teacher.load_state_dict(ckpt.get('teacher', ckpt))
            for p in self.gram_teacher.parameters():
                p.requires_grad_(False)
            print(f"[DINOv3] Gram teacher loaded from {self.gram_ckpt}")
        else:
            self.gram_teacher = copy.deepcopy(self.teacher)
            print("[DINOv3] No gram teacher checkpoint — using current teacher as gram anchor")

    def run_iteration(self, batch, do_backprop=True, run_online_evaluation=False):
        # Run standard DINOv3 iteration
        loss_val = super().run_iteration(batch, do_backprop=False,
                                          run_online_evaluation=False)

        if self.current_epoch >= self.gram_start_ep:
            data = batch['data']
            if isinstance(data, (list, tuple)):
                data = data[0]
            data = data.to(self.device, non_blocking=True)

            with autocast(self.device.type, enabled=True):
                s_out = self._get_tokens(self.network, data)[:, 1:]      # patch tokens
                with torch.no_grad():
                    g_out = self._get_tokens(self.gram_teacher, data)[:, 1:]

                l_gram = self.gram_loss(s_out, g_out, img_level=True)
                loss   = torch.tensor(loss_val, device=self.device) + self.lambda_gram * l_gram

            if do_backprop:
                self.optimizer.zero_grad(set_to_none=True)
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.network.parameters()) +
                    list(self.s_cls_head.parameters()) +
                    list(self.s_patch_head.parameters()), 3.0)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()

                ema_update(self.network, self.teacher, 0.996)
                ema_update(self.s_cls_head, self.t_cls_head, 0.996)
                ema_update(self.s_patch_head, self.t_patch_head, 0.996)

            return loss.detach().cpu().numpy()
        return loss_val
