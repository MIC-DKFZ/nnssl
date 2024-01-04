import numpy as np
import torch
from torch._C import device
from nnssl.architectures.build_architecture import build_network_architecture
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.training.loss.mse_loss import MSELoss
from nnssl.training.nnsslTrainer.AbstractMAETrainer import AbstractMAETrainer
from torch import device, nn


def create_blocky_mask(tensor_size, block_size, sparsity_factor: float = 0.75) -> torch.Tensor:
    """
    Create a binary mask for a tensor by creating a smaller mask and repeating it.

    :param tensor_size: Tuple of the dimensions of the tensor (height, width, depth).
    :param block_size: Size of the block to be masked (set to 0) in the smaller mask.
    :return: A binary mask tensor.
    """
    # Calculate the size of the smaller mask
    small_mask_size = tuple(size // block_size for size in tensor_size)

    # Create the smaller mask
    flat_mask = torch.ones(np.prod(small_mask_size))
    n_masked = int(sparsity_factor * flat_mask.shape[0])
    mask_indices = torch.randperm(flat_mask.shape[0])[:n_masked]
    flat_mask[mask_indices] = 0
    small_mask = torch.reshape(flat_mask, small_mask_size)
    large_mask = torch.repeat_interleave(small_mask, 4, dim=0).repeat_interleave(4, dim=1).repeat_interleave(4, dim=2)

    return large_mask


class BaseMAETrainer(AbstractMAETrainer):
    @staticmethod
    def mask_creation(batch_size: int, patch_size: tuple[int, int, int], mask_percentage: float) -> torch.Tensor:
        """
        Creates a masking tensor with 1s (indicating no masking) and 0s (indicating masking).
        The mask has to be of same size like the input data (batch_size, 1, x, y, z).

        :param patch_shape: The 3D shape information for the masking patch.
        :param mask_percentage: percentage of the patch that should be masked
        :param min_mask_block_size: minimum size of the blocks that should be masked
        :return:
        """

        block_size = 4
        sparsity_factor = 0.75
        mask = [create_blocky_mask(patch_size, block_size, sparsity_factor) for _ in range(batch_size)]
        mask = torch.stack(mask)[:, None, ...]  # Add channel dimension
        return mask
    
    


class DummyMAETrainer(BaseMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: device = ...,
    ):
        super().__init__(plan, configuration_name, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 2  # Just do two epochs to test if writing also works as intended.
