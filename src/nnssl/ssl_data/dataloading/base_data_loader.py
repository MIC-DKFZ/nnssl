from abc import ABC, abstractmethod
from typing import Union, Tuple

from batchgenerators.dataloading.data_loader import DataLoader
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *

from nnssl.data.dataloading.dataset import nnSSLDatasetBlosc2


class nnsslDataLoaderBase(DataLoader, ABC):

    def __init__(
        self,
        data: nnSSLDatasetBlosc2,
        batch_size: int,
        patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
        final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
        sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
        pad_sides: Union[List[int], Tuple[int, ...], np.ndarray] = None,
    ):
        super().__init__(data, batch_size, 1, None, True, False, True, sampling_probabilities)

        assert isinstance(data, nnSSLDatasetBlosc2), "nnSSLDataLoaderBase only supports nnsslDatasets."
        self.indices = list(data.image_identifiers)

        self._data: nnSSLDatasetBlosc2  # Set in the super class
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        # self.list_of_keys = list(self._data.keys())
        # need_to_pad denotes by how much we need to pad the data so that if we sample a patch of size final_patch_size
        # (which is what the network will get) these patches will also cover the border of the images
        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.num_channels = None
        self.pad_sides = pad_sides
        self.data_shape = self.determine_shapes()
        self.sampling_probabilities = sampling_probabilities

    def determine_shapes(self):
        # load one case
        data, _, _, _ = self._data[self.indices[0]]
        num_color_channels = data.shape[0]
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        return data_shape

    def get_bbox(
        self,
        data_shape: np.ndarray
    ):
        """Originally used to do probabilistic oversampling of foreground patches. We don't have foreground here though, so"""
        # in dataloader 2d we need to select the slice prior to this and also modify the class_locations to only have
        # locations for the given slice
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        for d in range(dim):
            # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            # always
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
        # define what the upper and lower bound can be to then sample form them with np.random.randint
        lbs = [-need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]

        # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
        # at least one of the foreground classes in the patch
        bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]

        return bbox_lbs, bbox_ubs

    @abstractmethod
    def generate_train_batch(self):
        """
        This is the function that has to be implemented to load data.
        It originally is defined in the DataLoader of Batchgenerators.
        Putting it here for easier readability."""
        pass
