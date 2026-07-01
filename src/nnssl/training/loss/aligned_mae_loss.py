from typing import Literal
from functools import lru_cache

import torch
import torch.nn.functional as F
from einops import rearrange

from torch import nn

from nnssl.training.loss.mse_loss import MAEMSELoss


# Correlated mask with extra positive offsets when using_teacher = True
@lru_cache(maxsize=5)
def _get_correlated_mask(b, device, using_teacher, verbose=False):
    eye = torch.eye(2 * b, device=device, dtype=torch.uint8)
    shifted = eye.roll(-b, dims=1)
    mask = eye + shifted

    if using_teacher:
        l_pos = eye.roll(-b // 2, dims=1)
        r_pos = eye.roll(b // 2, dims=1)
        mask = mask + l_pos + r_pos

    mask = (1 - mask).bool()  # invert to get negatives

    if verbose:
        import matplotlib.pyplot as plt

        plt.imshow(mask.detach().cpu().numpy(), interpolation="nearest")
        plt.title("Correlated Mask")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    return mask


class NTXentLoss(nn.Module):
    def __init__(
        self,
        temperature=0.5,
        similarity_function: Literal["cosine", "dot"] = "cosine",
        using_teacher=False,
    ):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.using_teacher = using_teacher
        self.similarity_function = similarity_function

    def _similarity(self, x, y):
        return x @ y.T  # (N, N) similarity

    def get_logits(self, zis, zjs):
        b = zis.size(0)
        device = zis.device

        # Concatenate all representations: [zjs, zis]
        reps = torch.cat([zjs, zis], dim=0)  # (2B, D)

        # Compute similarity matrix (2B x 2B)
        sim = self._similarity(reps, reps)  # full similarity, (2B, 2B)

        # Create the correlated mask
        mask = _get_correlated_mask(b, device, self.using_teacher)  # (2B, 2B)

        # Select positives: off-diagonal at offset ±b
        l_pos = torch.diag(sim, b)
        r_pos = torch.diag(sim, -b)
        positives = torch.cat([l_pos, r_pos]).view(2 * b, 1)  # (2B, 1)

        # Select negatives using the mask
        negatives = sim[mask].view(2 * b, -1)  # (2B, 2B - num_positives)

        logits = torch.cat([positives, negatives], dim=1)  # (2B, 1 + negs)
        logits /= self.temperature

        labels = torch.zeros(
            2 * b, dtype=torch.long, device=device
        )  # correct class is always index 0

        return logits, labels

    def forward(self, zis, zjs):
        b = zis.size(0)
        logits, labels = self.get_logits(zis, zjs)
        loss = self.criterion(logits, labels)
        accuracy = (logits.argmax(dim=1) == 0).float().mean().item()
        return loss / (2 * b), accuracy


class AlignedMAELoss(torch.nn.Module):

    def __init__(
        self,
        device,
        out_size=7,
        sampling_ratio=2,
        recon_weight=1.0,
        fg_cos_weight=0.5,
        ntxent_weight=0.1,
        fine_grained_contrastive: bool = False,
        fine_grained_cosine_regression: bool = False,
    ):
        """
        Initialize the KVConsisConLoss with the given parameters.

        Args:
            device (torch.device): The device to run the loss on.
            out_size (int or tuple[int, int, int]): The output size for the aligned latents.
            sampling_ratio (int): The ratio for sampling the output size.
            recon_weight (float): Weight for the reconstruction loss.
            fg_cos_weight (float): Weight for the finegrained cosine similarity loss.
            ntxent_weight (float): Weight for the NT-Xent loss.
            fine_grained_contrastive (bool): Whether to use fine-grained contrastive loss.
        """
        super(AlignedMAELoss, self).__init__()

        self.mse_loss = MAEMSELoss()
        self.huber = torch.nn.HuberLoss(reduction="none")
        self.fine_grained_contrastive = (
            fine_grained_contrastive  # whether to use fine-grained contrastive loss
        )
        self.fine_grained_cosine_regression = fine_grained_cosine_regression

        self.recon_key = "recon"
        self.proj_key = "proj"
        self.latent_key = "proj"
        self.image_latent_key = "image_latent"

        self.contrastive_loss = NTXentLoss(
            temperature=0.5, similarity_function="cosine", using_teacher=True
        )

        self.recon_weight = recon_weight
        self.fg_cos_weight = fg_cos_weight
        self.ntxent_weight = ntxent_weight

        # create a grid for resampling later
        # ── determine output resolution ──────────────────────────────────────────────
        if isinstance(out_size, int):
            D_out = H_out = W_out = out_size
        else:
            D_out, H_out, W_out = out_size
        self.D_out, self.H_out, self.W_out = D_out, H_out, W_out

        # ── build a base grid in [-1, 1] ────────────────────────────────────────────
        z_lin = torch.linspace(-1, 1, sampling_ratio * D_out, device=device)
        y_lin = torch.linspace(-1, 1, sampling_ratio * H_out, device=device)
        x_lin = torch.linspace(-1, 1, sampling_ratio * W_out, device=device)
        zz, yy, xx = torch.meshgrid(
            z_lin, y_lin, x_lin, indexing="ij"
        )  # (D_out, H_out, W_out)
        self.base_grid = torch.stack((xx, yy, zz), dim=-1).unsqueeze(
            0
        )  # (1, D_out, H_out, W_out, 3)

    def align_views(
        self,
        latents: torch.Tensor,
        rel_bboxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aligns the latents based on the relative bounding boxes.

        Args:
            latents (torch.Tensor): The latent representations [b, c, x_p, y_p, z_p].
            rel_bboxes (torch.Tensor): The relative bounding boxes. [b, 6] where each row is (x1, y1, z1, x2, y2, z2)
                and the values are in the range [0, 1].

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Aligned latents and bounding boxes.
        """
        B = latents.shape[0]

        # ── prepare per‑sample scale & shift ───────────────────────────────────────
        x1, y1, z1, x2, y2, z2 = rel_bboxes.unbind(dim=-1)  # each (B,)
        # centre in [0,1], size in [0,1]
        cx, cy, cz = (x1 + x2) * 0.5, (y1 + y2) * 0.5, (z1 + z2) * 0.5
        sx, sy, sz = (x2 - x1), (y2 - y1), (z2 - z1)

        # convert to shift / scale for [-1,1] space
        shift = torch.stack((2 * cx - 1, 2 * cy - 1, 2 * cz - 1), dim=-1)  # (B, 3)
        scale = torch.stack((sx, sy, sz), dim=-1)  # (B, 3)

        shift = shift.view(B, 1, 1, 1, 3)
        scale = scale.view(B, 1, 1, 1, 3)

        # ── produce the sampling grid ──────────────────────────────────────────────
        sampling_grid = self.base_grid * scale + shift  # (B, D_out, H_out, W_out, 3)

        # ── trilinear ROI‑align via grid_sample ────────────────────────────────────
        aligned_latents = F.grid_sample(
            latents,
            sampling_grid,
            mode="bilinear",  # when 5d input, "bilinear" is equivalent to "trilinear" internally
            padding_mode="border",
            align_corners=True,
        )

        # ── adaptive pool to the ouput size ────────────────────────────────────────
        aligned_latents = F.adaptive_avg_pool3d(
            aligned_latents, (self.W_out, self.H_out, self.D_out)
        )

        return aligned_latents

    def forward(
        self,
        model_output: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        gt_recon: torch.Tensor,
        rel_bboxes: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for the KVConsisConLoss.

        Args:
            model_output (dict[str, torch.Tensor]): The output from the model.
            target (dict[str, torch.Tensor]): The target values.
            gt_recon (torch.Tensor): Ground truth reconstruction.
            abs_bboxes (torch.Tensor): Relative bounding boxes.
            mask (torch.Tensor): Mask to apply to the loss.

        Returns:
            torch.Tensor: The computed loss value.
        """
        # Compute the consistency loss
        eps = torch.finfo(model_output[self.recon_key].dtype).eps

        recon_loss_huber = self.huber(model_output[self.recon_key], gt_recon)
        recon_loss_huber = torch.sum(recon_loss_huber * (1 - mask)) / (
            torch.sum((1 - mask)) + eps
        )

        recon_loss_mse = self.mse_loss(model_output[self.recon_key], gt_recon, mask)

        # chunk the latents and compute the consistency loss
        pred_latents_fg = model_output[self.proj_key]
        tgt_latents_fg = target[self.latent_key].detach()

        var_denom = 1 / target[self.proj_key].shape[1]
        cw_std = torch.std(
            F.normalize(target[self.proj_key].detach(), dim=1, eps=eps),
            dim=(0, 2, 3, 4),
        ).mean()
        cw_std = cw_std / (var_denom**0.5)

        # if latents is 5d tensor, i.e. [b, c, x_p, y_p, z_p], we need to align them for better consistency
        if pred_latents_fg.ndim == 5:
            pred_latents_fg = self.align_views(pred_latents_fg, rel_bboxes)
            tgt_latents_fg = self.align_views(tgt_latents_fg, rel_bboxes)

        b = pred_latents_fg.shape[0] // 2
        # swap the latents. the num_views is hardcoded to 2 for this method
        tgt_latents_fg = tgt_latents_fg.roll(b, 0)

        pred_latents_fg, tgt_latents_fg = F.normalize(
            pred_latents_fg, dim=1, eps=eps
        ), F.normalize(tgt_latents_fg, dim=1, eps=eps)

        if self.fine_grained_contrastive:
            fg_cos_reg, var = self.contrastive_loss(
                rearrange(pred_latents_fg, "b c x y z -> (b x y z) c"),
                rearrange(
                    tgt_latents_fg, "b c x y z -> (b x y z) c"
                ),  # swapped assignment already done
            )
        elif self.fine_grained_cosine_regression:
            fg_cos_reg = (
                2 - 2 * (pred_latents_fg * tgt_latents_fg).sum(dim=1).mean()
            )  # already normalized
            var = (
                torch.var(pred_latents_fg, dim=(0, 2, 3, 4), unbiased=False).mean()
                / var_denom
            )
        else:
            # Flatten into [B, N, C]
            _x_p = rearrange(pred_latents_fg, "b c x y z -> b (x y z) c")  # [B, N, C]
            _y_p = rearrange(tgt_latents_fg, "b c x y z -> b (x y z) c")  # [B, N, C]

            # Compute Gram matrices for all batches in parallel
            G_x = torch.bmm(_x_p.transpose(1, 2), _x_p)  # [B, C, C]
            G_y = torch.bmm(_y_p.transpose(1, 2), _y_p)  # [B, C, C]

            # Compute Gram regularization (mean squared difference)
            fg_cos_reg = torch.mean((G_x - G_y) ** 2)

            var = (
                torch.var(pred_latents_fg, dim=(0, 2, 3, 4), unbiased=False).mean()
                / var_denom
            )

        pred_latents_aa, tgt_latents_aa = (
            model_output[self.image_latent_key],
            target[self.image_latent_key].detach(),
        )
        tgt_latents_aa = tgt_latents_aa.roll(b, 0)

        contrastive_loss, acc = self.contrastive_loss(
            F.normalize(pred_latents_aa, dim=1, eps=eps),
            F.normalize(
                tgt_latents_aa.detach(), dim=1, eps=eps
            ),  # already swapped assignments
        )

        loss = (
            self.recon_weight * recon_loss_huber
            + self.fg_cos_weight * fg_cos_reg
            + self.ntxent_weight * contrastive_loss
        )

        return {
            "loss": loss,
            "huber": recon_loss_huber,
            "mse": recon_loss_mse,
            "cw_std": cw_std,
            "ntxent": contrastive_loss,
            "acc": torch.tensor(acc, dtype=torch.float, device=loss.device),
            "fg_cos_reg": fg_cos_reg,
            "var": var,
        }


if __name__ == "__main__":
    # _get_correlated_mask(4 * 5 * 5 * 5, torch.device("cuda"), using_teacher=True, verbose=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _model_output = {
        "recon": torch.randn(8, 1, 64, 64, 64, requires_grad=True, device=device),
        "proj": torch.randn(8, 2048, 16, 16, 16, requires_grad=True, device=device),
        "image_latent": torch.randn(8, 2048, requires_grad=True, device=device),
    }

    _target = {
        "proj": torch.randn(8, 2048, 20, 20, 20, device=device),
        "image_latent": torch.randn(8, 2048, device=device),
    }

    _gt_recon = torch.randn(
        8, 1, 64, 64, 64, device=device
    )  # Ground truth reconstruction

    _rel_bboxes = torch.tensor(
        [
            [0.1, 0.1, 0.1, 0.9, 0.9, 0.9],
            [0.2, 0.2, 0.2, 0.8, 0.8, 0.8],
            [0.3, 0.3, 0.3, 0.7, 0.7, 0.7],
            [0.4, 0.4, 0.4, 0.6, 0.6, 0.6],
            [0.1, 0.1, 0.1, 0.9, 0.9, 0.9],
            [0.2, 0.2, 0.2, 0.8, 0.8, 0.8],
            [0.3, 0.3, 0.3, 0.7, 0.7, 0.7],
            [0.4, 0.4, 0.4, 0.6, 0.6, 0.6],
        ],
        device=device,
    )  # Example relative bounding boxes
    _mask = torch.randint(
        0, 2, (8, 1, 64, 64, 64), device=device
    )  # Random mask for the example

    loss_fn = AlignedMAELoss(
        device,
        out_size=5,
        fine_grained_contrastive=False,
        fine_grained_cosine_regression=False,
        recon_weight=1.0,
        fg_cos_weight=0.5,
        ntxent_weight=0.1,
    )
    loss_fn.train(True)

    _loss_output = loss_fn(
        model_output=_model_output,
        target=_target,
        gt_recon=_gt_recon,
        rel_bboxes=_rel_bboxes,
        mask=_mask,
    )
    print(_loss_output)
