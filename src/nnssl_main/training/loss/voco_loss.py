import torch
from torch import nn
import torch.nn.functional as F


class VoCoLoss(nn.Module):
    """
    input must be logits, not probabilities!
    """

    def __init__(self, pred_weight=1, reg_weight=1):
        super(VoCoLoss, self).__init__()
        self.pred_weight = pred_weight
        self.reg_weight = reg_weight

    def prediction_loss(self, base_embeddings, target_embeddings, gt_overlaps) -> float:
        """
        This is the loss from EQ 3 from https://arxiv.org/pdf/2402.17300.
        It calculates the cosine similarity of representations with the GT being the volumetric overlap.
        """
        # We don't backprop through the base embeddings only the target embeddings.
        base_embeddings_de = base_embeddings.detach()
        pred_similarity = F.cosine_similarity(
            base_embeddings_de[:, None, :, :], target_embeddings[:, :, None, :], dim=-1
        )
        # Now one may think that EQ 1 may imply that this is the logits already...
        # But the implementation https://github.com/Luffy03/VoCo/blob/1ae3fc9ece19031a74c54cb12c1c8f01715d7f90/models/voco_head.py#L218
        # Clearly shows they apply RELU first...  #QualityDescription
        logits = F.relu(pred_similarity)

        # But wait ... There is more!
        # EQ3 from the paper clearly states that the SAME LOSS IS APPLIED TO ALL CROPS!
        # Guess what? ... https://github.com/Luffy03/VoCo/blob/1ae3fc9ece19031a74c54cb12c1c8f01715d7f90/models/voco_head.py#L249
        # Let's just apply a different loss for cases where GT overlap is non-zero than for the zero cases...
        # ... Thanks a lot for the headache!
        #    This would have been the code if it wasn't wrongly descibed in the paper...
        #       sim_dist = torch.abs(gt_overlaps - logits)
        #       N = sim_dist.shape[-1] * sim_dist.shape[-2]
        #       ce_loss = - torch.sum(torch.log(1 - sim_dist), dim=(1, 2)) / N
        pos_dist = torch.abs(gt_overlaps - logits)
        pos_pos = torch.where(gt_overlaps > 0, torch.ones_like(gt_overlaps), torch.zeros_like(gt_overlaps))
        # pos_loss = ((-torch.log(1 - pos_dist + 1e-6)) * pos_pos).sum() / (pos_pos.sum())
        pos_loss = ((-torch.log(1 - pos_dist + 1e-6)) * gt_overlaps).sum() / (gt_overlaps.sum())    # use overlap factor
        neg_loss = ((logits**2) * (1 - pos_pos)).sum() / (1 - pos_pos + 1e-6).sum()

        # Best thing is: They are not even consistent in their old and new code version.
        #   In the new version, they weight the pos_loss by their overlap factor
        #   In the old version, the weight is applied identically to both losses.

        # Aggregate per crop then average across batch samples.
        l_pred = pos_loss + neg_loss
        return l_pred

    def regularization_loss(self, base_embeddings: torch.Tensor) -> float:
        """
        This is the loss of EQ 5 from https://arxiv.org/pdf/2402.17300
        It enforces that non-overlapping base embeddings are orthogonal to each other.
        """
        inter_crop_similarity = F.cosine_similarity(
            base_embeddings[:, None, :, :],
            base_embeddings[:, :, None, :],
            dim=-1,
        )
        # Again ReLu'd because why not..
        inter_crop_sim_relu = F.relu(inter_crop_similarity)

        up_tri = torch.ones(
            inter_crop_sim_relu.shape[-2], inter_crop_sim_relu.shape[-1], device=inter_crop_sim_relu.device
        ).triu(diagonal=1)[None, ...]

        #
        upper_triangular = up_tri * inter_crop_sim_relu
        N = upper_triangular.shape[-1]
        # Aggregate per image cluster then average across batch samples.
        l_reg = torch.mean(torch.sum(upper_triangular, dim=(-2, -1)) * 2 / (N * (N - 1)))
        return l_reg

    def forward(
        self,
        base_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        gt_overlaps: torch.Tensor,
    ):
        """Implements the joint VoCo Loss (EQ7) from https://arxiv.org/pdf/2402.17300."""
        pred_loss = self.prediction_loss(base_embeddings, target_embeddings, gt_overlaps)
        reg_loss = self.regularization_loss(base_embeddings)
        final_loss = self.pred_weight * pred_loss + self.reg_weight * reg_loss
        return final_loss
