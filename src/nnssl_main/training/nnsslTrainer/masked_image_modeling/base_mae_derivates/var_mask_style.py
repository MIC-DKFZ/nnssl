import random
import numpy as np
import torch

from nnssl.experiment_planning.experiment_planners.plan import Plan
from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import BaseMAETrainer


def create_blocky_mask(tensor_size, block_size, sparsity_factor=0.75, rng_seed: None | int = None) -> torch.Tensor:
    """
    Create the smallest binary mask for the encoder by choosing a percentage of pixels at that resolution..

    :param tensor_size: Tuple of the dimensions of the tensor (height, width, depth).
    :param block_size: Size of the block to be masked (set to 0) in the smaller mask.
    :return: A binary mask tensor.
    """
    # Calculate the size of the smaller mask
    small_mask_size = tuple(size // block_size for size in tensor_size)

    # Create the smaller mask
    flat_mask = torch.ones(np.prod(small_mask_size))
    n_masked = int(sparsity_factor * flat_mask.shape[0])
    if rng_seed is None:
        mask_indices = torch.randperm(flat_mask.shape[0])[:n_masked]
    else:
        gen = torch.Generator.manual_seed(rng_seed)
        mask_indices = torch.randperm(flat_mask.shape[0], generator=gen)[:n_masked]
    flat_mask[mask_indices] = 0
    small_mask = torch.reshape(flat_mask, small_mask_size)
    return small_mask


def create_grid_mask(tensor_size, block_size, sparsity_factor=0.75, rng_seed: None | int = None) -> torch.Tensor:
    """
    Create a regular grid of block masks for the encoder by choosing a percentage of pixels at that resolution..

    :param tensor_size: Tuple of the dimensions of the tensor (height, width, depth).
    :param block_size: Size of the block to be masked (set to 0) in the smaller mask.
    :return: A binary mask tensor.
    """
    # Create the smaller mask
    # Create a grid of blocks
    # Since this is 3D, we only have 8 possible mask ratios.
    # 0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100
    # So we map the sparse factor to the closest ratio
    repeat_factors = [ts // (2 * block_size) for ts in tensor_size]
    closest_ratio = round(sparsity_factor * 8) / 8
    # Depending on the ratio, we create the grid

    mask_kernel = torch.ones((2, 2, 2))
    if closest_ratio == 0 or closest_ratio == 1:
        raise ValueError("The sparsity factor is too low or too high for the grid mask.")
    if closest_ratio >= 0.125:
        mask_kernel[0, 0, 0] = 0
    if closest_ratio >= 0.25:
        mask_kernel[1, 0, 0] = 0
    if closest_ratio >= 0.375:
        mask_kernel[0, 1, 0] = 0
    if closest_ratio >= 0.5:
        mask_kernel[1, 1, 0] = 0
    if closest_ratio >= 0.625:
        mask_kernel[0, 0, 1] = 0
    if closest_ratio >= 0.75:
        mask_kernel[1, 0, 1] = 0
    if closest_ratio >= 0.875:
        mask_kernel[0, 1, 1] = 0

    full_mask_kernel = (
        mask_kernel.repeat_interleave(block_size, dim=0)
        .repeat_interleave(block_size, dim=1)
        .repeat_interleave(block_size, dim=2)
    )

    # Repeat for full tensor mask - but not interleaved
    full_mask = full_mask_kernel.repeat(repeat_factors)

    return full_mask


def create_slice_mask(tensor_size, sparsity_factor=0.75, rng_seed: None | int = None) -> torch.Tensor:
    """Create masking based of removing indices and slices."""

    potential_slices = tensor_size
    total_slice_ids = [(cnt, j) for cnt, i in enumerate(potential_slices) for j in range(i)]

    n_slices = int(sparsity_factor * len(total_slice_ids) / 3)
    if rng_seed is None:
        slice_indices = random.sample(total_slice_ids, n_slices)
    else:
        random.seed(rng_seed)
        slice_indices = random.sample(total_slice_ids, n_slices)

    mask = torch.ones(tensor_size)
    slice_by_axis = [[j for ax, j in slice_indices if ax == i] for i in [0, 1, 2]]
    mask[slice_by_axis[0]] = 0
    mask[:, slice_by_axis[1]] = 0
    mask[:, :, slice_by_axis[2]] = 0
    return mask


def create_same_dim_slice_mask(tensor_size, sparsity_factor=0.75, rng_seed: None | int = None) -> torch.Tensor:
    """Create masking based of removing indices and slices."""

    slice_of_choice = random.choice([0, 1, 2])
    total_slice_ids = [(slice_of_choice, j) for j in range(tensor_size[slice_of_choice])]

    n_slices = int(sparsity_factor * len(total_slice_ids))
    if rng_seed is None:
        slice_indices = random.sample(total_slice_ids, n_slices)
    else:
        random.seed(rng_seed)
        slice_indices = random.sample(total_slice_ids, n_slices)

    mask = torch.ones(tensor_size)
    slice_by_axis = [[j for ax, j in slice_indices if ax == i] for i in [0, 1, 2]]
    mask[slice_by_axis[0]] = 0
    mask[:, slice_by_axis[1]] = 0
    mask[:, :, slice_by_axis[2]] = 0
    return mask


class GridMaskMAE(BaseMAETrainer):

    @staticmethod
    def mask_creation(
        batch_size: int, patch_size: tuple[int, int, int], mask_percentage: float, rng_seed: int | None = None
    ) -> torch.Tensor:
        """
        Creates a masking tensor with 1s (indicating no masking) and 0s (indicating masking).
        The mask has to be of same size like the input data (batch_size, 1, x, y, z).

        :param patch_shape: The 3D shape information for the masking patch.
        :param mask_percentage: percentage of the patch that should be masked
        :param min_mask_block_size: minimum size of the blocks that should be masked
        :return:
        """

        block_size = 16
        sparsity_factor = mask_percentage
        mask = [create_grid_mask(patch_size, block_size, sparsity_factor) for _ in range(batch_size)]
        mask = torch.stack(mask)[:, None, ...]  # Add channel dimension
        return mask


class SliceMaskMAE(BaseMAETrainer):

    @staticmethod
    def mask_creation(
        batch_size: int, patch_size: tuple[int, int, int], mask_percentage: float, rng_seed: int | None = None
    ) -> torch.Tensor:
        """
        Creates a masking tensor with 1s (indicating no masking) and 0s (indicating masking).
        The mask has to be of same size like the input data (batch_size, 1, x, y, z).

        :param patch_shape: The 3D shape information for the masking patch.
        :param mask_percentage: percentage of the patch that should be masked
        :param min_mask_block_size: minimum size of the blocks that should be masked
        :return:
        """

        mask = [create_slice_mask(patch_size, mask_percentage) for _ in range(batch_size)]
        mask = torch.stack(mask)[:, None, ...]  # Add channel dimension
        return mask


class SameSliceMaskMAE(BaseMAETrainer):

    @staticmethod
    def mask_creation(
        batch_size: int, patch_size: tuple[int, int, int], mask_percentage: float, rng_seed: int | None = None
    ) -> torch.Tensor:
        """
        Creates a masking tensor with 1s (indicating no masking) and 0s (indicating masking).
        The mask has to be of same size like the input data (batch_size, 1, x, y, z).

        :param patch_shape: The 3D shape information for the masking patch.
        :param mask_percentage: percentage of the patch that should be masked
        :param min_mask_block_size: minimum size of the blocks that should be masked
        :return:
        """

        mask = [create_same_dim_slice_mask(patch_size, mask_percentage) for _ in range(batch_size)]
        mask = torch.stack(mask)[:, None, ...]  # Add channel dimension
        return mask


class SliceMaskMAETrainer_BS8_ep1000_mask90(SliceMaskMAE):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 8
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.mask_percentage: float = 0.90


class SliceMaskMAETrainer_BS8_ep1000_mask75(SliceMaskMAE):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 8
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.mask_percentage: float = 0.75


class SliceMaskMAETrainer_BS8_ep1000_mask60(SliceMaskMAE):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 8
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.mask_percentage: float = 0.60


class SliceMaskMAETrainer_BS8_ep1000_mask45(SliceMaskMAE):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 8
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.mask_percentage: float = 0.45


class SameSliceMaskMAETrainer_BS8_ep1000_mask60(SameSliceMaskMAE):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 8
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.mask_percentage: float = 0.60


class SameSliceMaskMAETrainer_BS8_ep1000_mask45(SameSliceMaskMAE):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 8
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.mask_percentage: float = 0.45


class SameSliceMaskMAETrainer_BS8_ep1000_mask75(SameSliceMaskMAE):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 8
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.mask_percentage: float = 0.75


class SameSliceMaskMAETrainer_BS8_ep1000_mask90(SameSliceMaskMAE):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 8
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.mask_percentage: float = 0.90


class GridMaskMAE_BS8_ep1000_mask60(GridMaskMAE):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 8
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.mask_percentage: float = 0.60


class GridMaskMAE_BS8_ep1000_mask75(GridMaskMAE):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 8
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.mask_percentage: float = 0.75


class GridMaskMAE_BS8_ep1000_mask90(GridMaskMAE):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 8
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.mask_percentage: float = 0.90


class GridMaskMAE_BS8_ep1000_mask45(GridMaskMAE):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 8
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.mask_percentage: float = 0.45


class GridMaskMAE_BS1_mask75(GridMaskMAE):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 1
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.mask_percentage: float = 0.75


class SliceMaskMAE_BS1_mask75(SliceMaskMAE):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 1
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.mask_percentage: float = 0.75


class SameSliceMaskMAE_BS1_mask75(SameSliceMaskMAE):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 1
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.mask_percentage: float = 0.75
