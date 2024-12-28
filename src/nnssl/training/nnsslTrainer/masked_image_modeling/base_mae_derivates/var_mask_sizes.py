import torch
from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import BaseMAETrainer_BS8_1000ep, create_blocky_mask


class BaseMAETrainer_BS8_ep1000_maskblock2(BaseMAETrainer_BS8_1000ep):

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

        block_size = 2
        sparsity_factor = mask_percentage
        mask = [create_blocky_mask(patch_size, block_size, sparsity_factor) for _ in range(batch_size)]
        mask = torch.stack(mask)[:, None, ...]  # Add channel dimension
        return mask


class BaseMAETrainer_BS8_ep1000_maskblock4(BaseMAETrainer_BS8_1000ep):

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

        block_size = 4
        sparsity_factor = mask_percentage
        mask = [create_blocky_mask(patch_size, block_size, sparsity_factor) for _ in range(batch_size)]
        mask = torch.stack(mask)[:, None, ...]  # Add channel dimension
        return mask


class BaseMAETrainer_BS8_ep1000_maskblock8(BaseMAETrainer_BS8_1000ep):
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

        block_size = 8
        sparsity_factor = mask_percentage
        mask = [create_blocky_mask(patch_size, block_size, sparsity_factor) for _ in range(batch_size)]
        mask = torch.stack(mask)[:, None, ...]  # Add channel dimension
        return mask


class BaseMAETrainer_BS8_ep1000_maskblock32(BaseMAETrainer_BS8_1000ep):
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

        block_size = 32
        sparsity_factor = mask_percentage
        mask = [create_blocky_mask(patch_size, block_size, sparsity_factor) for _ in range(batch_size)]
        mask = torch.stack(mask)[:, None, ...]  # Add channel dimension
        return mask


class BaseMAETrainer_BS8_ep1000_maskblock64(BaseMAETrainer_BS8_1000ep):
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

        block_size = 64
        sparsity_factor = mask_percentage
        mask = [create_blocky_mask(patch_size, block_size, sparsity_factor) for _ in range(batch_size)]
        mask = torch.stack(mask)[:, None, ...]  # Add channel dimension
        return mask
