import numpy as np
import torch


def add_masks_of_size(existing_mask: torch.tensor,  mask_size: int, mask_percentage: float,no_overlap: bool=True):
    """ Add masks that (either overlap or don't overlap) to the image."""
    mask = existing_mask
    
    n_blocks = int(torch.prod(torch.tensor(patch_shape)) * mask_percentage) // (min_mask_block_size) ** 3
    block_size = max(min_mask_block_size, int(num_blocks ** (1 / 3)))
    mask_blocks = torch.randperm(torch.prod(torch.tensor(patch_shape)))[:num_blocks]
    for block in mask_blocks:
        indices = torch.unravel_index(block, patch_shape)
        mask[
            indices[0] : indices[0] + block_size,
            indices[1] : indices[1] + block_size,
            indices[2] : indices[2] + block_size,
        ] = 0
    return mask

def create_smlv_masks(patch_shape, mask_percentages: tuple[float, float, float, float] = , block_size: tuple[int, int, int, int] = [4, 8, 16, 32]):
    """ Mask the input image with a mask of the """
    mask = torch.ones(patch_shape)
    n_blocks = int(torch.prod(torch.tensor(patch_shape)) * mask_percentage) // (min_mask_block_size) ** 3
    block_size = max(min_mask_block_size, int(num_blocks ** (1 / 3)))
    mask_blocks = torch.randperm(torch.prod(torch.tensor(patch_shape)))[:num_blocks]
    for block in mask_blocks:
        indices = torch.unravel_index(block, patch_shape)
        mask[
            indices[0] : indices[0] + block_size,
            indices[1] : indices[1] + block_size,
            indices[2] : indices[2] + block_size,
        ] = 0
    return mask


if __name__ == "__main__":
    patch_shape = (32, 32, 32)
    mask_percentage = 0.5
    min_mask_block_size = 4
    mask = create_mask(patch_shape, mask_percentage, min_mask_block_size)
    print(mask)
