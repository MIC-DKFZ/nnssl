from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from random import randint


class StructuredPatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794 and https://arxiv.org/pdf/2208.07220
    """

    return_indices: torch.jit.Final[bool]

    def __init__(
        self,
        patchified_im_shape: Tuple[int, int, int],
        prob: float = 0.5,
        num_prefix_tokens: int = 1,
        ordered: bool = False,
        return_indices: bool = False,
    ):
        assert 0 <= prob < 1.0, "Probability must be in [0, 1)"
        assert len(patchified_im_shape) == 3, "Currently only 3d is supported"
        assert num_prefix_tokens == 0, "Currently only 0 prefix tokens are supported"

        super().__init__()
        self.patchified_im_shape = patchified_im_shape
        self.prob = prob
        self.ordered = ordered
        self.return_indices = return_indices
        self.num_prefix_tokens = num_prefix_tokens  # exclude CLS token (or other prefix tokens)

        total_seq_length = patchified_im_shape[0] * patchified_im_shape[1] * patchified_im_shape[2]
        wanted_remaining_tokens = round(total_seq_length * (1.0 - prob))
        tokens_per_x_slice = patchified_im_shape[1] * patchified_im_shape[2]
        tokens_per_y_slice = patchified_im_shape[0] * patchified_im_shape[2]
        tokens_per_z_slice = patchified_im_shape[0] * patchified_im_shape[1]
        token_per_dim = [tokens_per_x_slice, tokens_per_y_slice, tokens_per_z_slice]
        self.full_slices_to_keep_per_dim = [wanted_remaining_tokens // tpd for tpd in token_per_dim]  # X, Y, Z
        self.remaining_N_tokens_per_dim = [wanted_remaining_tokens % tpd for tpd in token_per_dim]  # X, Y, Z

    def determine_keep_indices(self, x: torch.Tensor, B: int):
        """Determines which indices to keep.
        This function checks the pre-flattened patch shape and then
        1. Picks one of the two/three dimensions
        2. Determines the amount of slices along the dimension to keep
        3. Randomly picks a starting point and then fills up the slices to the max
        4. all lines along a certain axis at a certain position.
        """
        # Randomly drop-out contiguous slices to maintain
        X, Y, Z = self.patchified_im_shape

        all_keep_indices = []
        for i in range(B):
            dim_chosen = randint(0, len(self.patchified_im_shape) - 1)
            n_slices_to_keep = self.full_slices_to_keep_per_dim[dim_chosen]
            leftover = self.remaining_N_tokens_per_dim[dim_chosen]
            # Keep one buffer slices at beginning and end
            start_slice = randint(1, self.patchified_im_shape[dim_chosen] - (n_slices_to_keep + 1))

            # ------------ Take previous or next slice for filling remainders ------------ #
            prev_for_remained = randint(0, 1)
            if prev_for_remained:
                next_slice = start_slice - 1
            else:
                next_slice = start_slice + n_slices_to_keep

            mask = torch.zeros(self.patchified_im_shape, device=x.device, dtype=torch.bool)
            if dim_chosen == 0:
                # Slices along x => for each i, set mask[i, :, :] = True
                mask[start_slice : (n_slices_to_keep + start_slice), :, :] = True

                if leftover > 0:
                    # We have Y*Z possible tokens in that slice
                    yz_permutation = torch.randperm(Y * Z, device=x.device)
                    leftover_indices = yz_permutation[:leftover]
                    # leftover_indices range in [0..(Y*Z-1)], unravel to (y,z)
                    leftover_y = leftover_indices // Z
                    leftover_z = leftover_indices % Z
                    mask[next_slice, leftover_y, leftover_z] = True

            elif dim_chosen == 1:
                # Slices along y => for each i, set mask[:, i, :] = True
                mask[:, start_slice : (n_slices_to_keep + start_slice), :] = True

                if leftover > 0:
                    xz_permutation = torch.randperm(X * Z, device=x.device)
                    leftover_indices = xz_permutation[:leftover]
                    leftover_x = leftover_indices // Z
                    leftover_z = leftover_indices % Z
                    mask[leftover_x, next_slice, leftover_z] = True

            else:
                # dim_chosen == 2
                # Slices along z => for each i, set mask[:, :, i] = True
                mask[:, :, start_slice : (n_slices_to_keep + start_slice)] = True

                if leftover > 0:
                    xy_permutation = torch.randperm(X * Y, device=x.device)
                    leftover_indices = xy_permutation[:leftover]
                    leftover_x = leftover_indices // Y
                    leftover_y = leftover_indices % Y
                    mask[leftover_x, leftover_y, next_slice] = True
            all_keep_indices.append(mask.view(-1).nonzero(as_tuple=False).squeeze())  # [K]

        return torch.stack(all_keep_indices, dim=0)

    def forward(self, x) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        if not self.training or self.prob == 0.0:
            if self.return_indices:
                return x, None
            return x

        if self.num_prefix_tokens:
            prefix_tokens, x = x[:, : self.num_prefix_tokens], x[:, self.num_prefix_tokens :]
        else:
            prefix_tokens = None

        keep_indices = self.determine_keep_indices(x, x.shape[0])
        if self.ordered:
            # NOTE does not need to maintain patch order in typical transformer use,
            # but possibly useful for debug / visualization
            keep_indices = keep_indices.sort(dim=-1)[0]
        x = x.gather(1, keep_indices.unsqueeze(-1).expand((-1, -1) + x.shape[2:]))

        if prefix_tokens is not None:
            x = torch.cat((prefix_tokens, x), dim=1)

        if self.return_indices:
            return x, keep_indices
        return x
