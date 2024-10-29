import os
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Union, Type, Tuple

import numpy as np
import blosc2

from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile, write_pickle, subfiles
from nnunetv2.configuration import default_num_processes
from nnunetv2.training.dataloading.utils import unpack_dataset
import math


class nnSSLBaseDataset(ABC):
    """
    Defines the interface
    """

    def __init__(self, folder: str, identifiers: List[str] = None, folder_with_segs_from_previous_stage: str = None):
        super().__init__()
        # print('loading dataset')
        if identifiers is None:
            identifiers = self.get_identifiers(folder)
        identifiers.sort()

        self.source_folder = folder
        self.folder_with_segs_from_previous_stage = folder_with_segs_from_previous_stage
        self.identifiers = identifiers

    def __getitem__(self, identifier):
        return self.load_case(identifier)

    @abstractmethod
    def load_case(self, identifier):
        pass

    @staticmethod
    @abstractmethod
    def save_case(data: np.ndarray, seg: np.ndarray, properties: dict, output_filename_truncated: str):
        pass

    @staticmethod
    @abstractmethod
    def get_identifiers(folder: str) -> List[str]:
        pass

    @staticmethod
    def unpack_dataset(
        folder: str, overwrite_existing: bool = False, num_processes: int = default_num_processes, verify: bool = True
    ):
        pass


class nnSSLDatasetNumpy(nnSSLBaseDataset):
    def load_case(self, identifier):
        raise NotImplementedError

    @staticmethod
    def save_case(data: np.ndarray, seg: np.ndarray, properties: dict, output_filename_truncated: str):
        raise NotImplementedError

    @staticmethod
    def get_identifiers(folder: str) -> List[str]:
        """
        returns all identifiers in the preprocessed data folder
        """
        case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith("npz")]
        return case_identifiers

    @staticmethod
    def unpack_dataset(
        folder: str, overwrite_existing: bool = False, num_processes: int = default_num_processes, verify: bool = True
    ):
        raise NotImplementedError


class nnSSLDatasetBlosc2(nnSSLBaseDataset):
    def __init__(self, folder: str, identifiers: List[str] = None, folder_with_segs_from_previous_stage: str = None):
        super().__init__(folder, identifiers, folder_with_segs_from_previous_stage)
        blosc2.set_nthreads(1)

    def __getitem__(self, identifier):
        return self.load_case(identifier)

    def load_case(self, identifier):
        dparams = {"nthreads": 1}
        data_b2nd_file = join(self.source_folder, identifier + ".b2nd")
        data = blosc2.open(urlpath=data_b2nd_file, mode="r", dparams=dparams, mmap_mode="r")

        anon_b2nd_file = join(self.source_folder, identifier + "__anon.b2nd")
        if isfile(anon_b2nd_file):
            anon = blosc2.open(urlpath=anon_b2nd_file, mode="r", dparams=dparams, mmap_mode="r")
        else:
            anon = None

        deface_b2nd_file = join(self.source_folder, identifier + "__deface.b2nd")
        if isfile(deface_b2nd_file):
            deface = blosc2.open(urlpath=deface_b2nd_file, mode="r", dparams=dparams, mmap_mode="r")
        else:
            deface = None

        properties = load_pickle(join(self.source_folder, identifier + ".pkl"))
        return data, anon, deface, properties

    @staticmethod
    def save_case(
        data: np.ndarray,
        anon_mask: np.ndarray | None,
        deface_mask: np.ndarray | None,
        properties: dict,
        output_filename_truncated: str,
        chunks=None,
        blocks=None,
        chunks_seg=None,
        blocks_seg=None,
        clevel: int = 8,
        codec=blosc2.Codec.ZSTD,
    ):
        blosc2.set_nthreads(1)
        if chunks_seg is None:
            chunks_seg = chunks
        if blocks_seg is None:
            blocks_seg = blocks

        cparams = {
            "codec": codec,
            # 'filters': [blosc2.Filter.SHUFFLE],
            # 'splitmode': blosc2.SplitMode.ALWAYS_SPLIT,
            "clevel": clevel,
        }
        blosc2.asarray(
            np.ascontiguousarray(data),
            urlpath=output_filename_truncated + ".b2nd",
            chunks=chunks,
            blocks=blocks,
            cparams=cparams,
            mmap_mode="w+",
        )

        if anon_mask is not None:
            blosc2.asarray(
                np.ascontiguousarray(anon_mask),
                urlpath=output_filename_truncated + "__anon.b2nd",
                chunks=chunks_seg,
                blocks=blocks_seg,
                cparams=cparams,
                mmap_mode="w+",
            )

        if deface_mask is not None:
            blosc2.asarray(
                np.ascontiguousarray(deface_mask),
                urlpath=output_filename_truncated + "__deface.b2nd",
                chunks=chunks_seg,
                blocks=blocks_seg,
                cparams=cparams,
                mmap_mode="w+",
            )

        write_pickle(properties, output_filename_truncated + ".pkl")

    @staticmethod
    def get_identifiers(folder: str) -> List[str]:
        """
        returns all identifiers in the preprocessed data folder
        """
        case_identifiers = [i[:-5] for i in os.listdir(folder) if i.endswith(".b2nd") and not i.endswith("_seg.b2nd")]
        return case_identifiers

    @staticmethod
    def unpack_dataset(
        folder: str, overwrite_existing: bool = False, num_processes: int = default_num_processes, verify: bool = True
    ):
        pass

    @staticmethod
    def comp_blosc2_params(
        image_size: Tuple[int, int, int, int],
        patch_size: Union[Tuple[int, int], Tuple[int, int, int]],
        bytes_per_pixel: int = 4,  # 4 byte are float32
        l1_cache_size_per_core_in_bytes=32768,  # 1 Kibibyte (KiB) = 2^10 Byte;  32 KiB = 32768 Byte
        l3_cache_size_per_core_in_bytes=1441792,
        # 1 Mibibyte (MiB) = 2^20 Byte = 1.048.576 Byte; 1.375MiB = 1441792 Byte
        safety_factor: float = 0.8,  # we dont will the caches to the brim. 0.8 means we target 80% of the caches
    ):
        """
        Computes a recommended block and chunk size for saving arrays with blosc v2.

        Bloscv2 NDIM doku: "Remember that having a second partition means that we have better flexibility to fit the
        different partitions at the different CPU cache levels; typically the first partition (aka chunks) should
        be made to fit in L3 cache, whereas the second partition (aka blocks) should rather fit in L2/L1 caches
        (depending on whether compression ratio or speed is desired)."
        (https://www.blosc.org/posts/blosc2-ndim-intro/)
        -> We are not 100% sure how to optimize for that. For now we try to fit the uncompressed block in L1. This
        might spill over into L2, which is fine in our books.

        Note: this is optimized for nnU-Net dataloading where each read operation is done by one core. We cannot use threading

        Cache default values computed based on old Intel 4110 CPU with 32K L1, 128K L2 and 1408K L3 cache per core.
        We cannot optimize further for more modern CPUs with more cache as the data will need be be read by the
        old ones as well.

        Args:
            patch_size: Image size, must be 4D (c, x, y, z). For 2D images, make x=1
            patch_size: Patch size, spatial dimensions only. So (x, y) or (x, y, z)
            bytes_per_pixel: Number of bytes per element. Example: float32 -> 4 bytes
            l1_cache_size_per_core_in_bytes: The size of the L1 cache per core in Bytes.
            l3_cache_size_per_core_in_bytes: The size of the L3 cache exclusively accessible by each core. Usually the global size of the L3 cache divided by the number of cores.

        Returns:
            The recommended block and the chunk size.
        """
        # Fabians code is ugly, but eh

        num_channels = image_size[0]
        if len(patch_size) == 2:
            patch_size = [1, *patch_size]
        patch_size = np.array(patch_size)
        block_size = np.array((num_channels, *[2 ** (max(0, math.floor(math.log2(i / 2)))) for i in patch_size]))

        # shrink the block size until it fits in L1
        estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel
        while estimated_nbytes_block > (l1_cache_size_per_core_in_bytes * safety_factor):
            # pick largest deviation from patch_size that is not 1
            axis_order = np.argsort(block_size[1:] / patch_size)[::-1]
            idx = 0
            picked_axis = axis_order[idx]
            while block_size[picked_axis + 1] == 1 or block_size[picked_axis + 1] == image_size[picked_axis + 1]:
                idx += 1
                picked_axis = axis_order[idx]
            # now reduce that axis to the next lowest power of 2
            block_size[picked_axis + 1] = 2 ** (max(0, math.floor(math.log2(block_size[picked_axis + 1] - 1))))
            block_size[picked_axis + 1] = min(block_size[picked_axis + 1], image_size[picked_axis + 1])
            estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel
            if all([i == j for i, j in zip(block_size, image_size)]):
                break

        # note: there is no use extending the chunk size to 3d when we have a 2d patch size! This would unnecessarily
        # load data into L3
        # now tile the blocks into chunks until we hit image_size or the l3 cache per core limit
        chunk_size = deepcopy(block_size)
        estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
        while estimated_nbytes_chunk < (l3_cache_size_per_core_in_bytes * safety_factor):
            # find axis that deviates from block_size the most
            axis_order = np.argsort(chunk_size[1:] / block_size[1:])
            idx = 0
            picked_axis = axis_order[idx]
            while chunk_size[picked_axis + 1] == image_size[picked_axis + 1] or patch_size[picked_axis] == 1:
                idx += 1
                picked_axis = axis_order[idx]
            chunk_size[picked_axis + 1] += block_size[picked_axis + 1]
            chunk_size[picked_axis + 1] = min(chunk_size[picked_axis + 1], image_size[picked_axis + 1])
            estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
            if patch_size[0] == 1:
                if all([i == j for i, j in zip(chunk_size[2:], image_size[2:])]):
                    break
            if all([i == j for i, j in zip(chunk_size, image_size)]):
                break
        # print(image_size, chunk_size, block_size)
        return tuple(block_size), tuple(chunk_size)


file_ending_dataset_mapping = {"npz": nnSSLDatasetNumpy, "b2nd": nnSSLDatasetBlosc2}


def infer_dataset_class(folder: str) -> Union[Type[nnSSLDatasetBlosc2], Type[nnSSLDatasetNumpy]]:
    file_endings = set([os.path.basename(i).split(".")[-1] for i in subfiles(folder, join=False)])
    if "pkl" in file_endings:
        file_endings.remove("pkl")
    if "npy" in file_endings:
        file_endings.remove("npy")
    assert len(file_endings) == 1, (
        f"Found more than one file ending in the folder {folder}. " f"Unable to infer nnUNetDataset variant!"
    )
    return file_ending_dataset_mapping[list(file_endings)[0]]
