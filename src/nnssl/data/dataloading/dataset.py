import os
from abc import ABC, abstractmethod
from copy import deepcopy
from random import choice
from typing import List, Union, Type, Tuple

import numpy as np
import blosc2

from batchgenerators.utilities.file_and_folder_operations import (
    join,
    load_pickle,
    isfile,
    write_pickle,
    subfiles,
)
import math

from nnssl.data.raw_dataset import Collection, Dataset, IndependentImage, Subject
from nnssl.data.nnsslFilter.abstract_filter import AbstractFilter


class nnSSLBaseDataset(ABC):
    """
    Defines the interface
    """

    def __init__(
        self,
        dataset_dir: str,
        collection: Collection,
        subject_identifiers: List[str] = None,
        iimg_filters: list[AbstractFilter] = None,
    ):
        """
        Receives a dataset object that is created by loading the `pretrain_data.json`.
        This dataset object holds all the necessary info to load the data from disk.

        The subject_identifier are additional infos that determine which subjects are included in this dataset - All others are discarded.
        """
        super().__init__()
        # print('loading dataset')

        self.dataset_dir: str = dataset_dir
        self.subject_identifiers = set(
            subject_identifiers
        )  # Make it a set for faster lookup
        self.iimg_filters = iimg_filters if iimg_filters is not None else []
        self.collection = deepcopy(collection)

        all_images: list[IndependentImage] = self.collection.to_independent_images()

        self.image_dataset: dict[str, IndependentImage] = {
            im.get_unique_id(): im
            for im in all_images
            if im.get_unique_subject_id() in self.subject_identifiers
            if all(iimg_filter(im) for iimg_filter in self.iimg_filters)
        }
        self.image_identifiers: list[str] = list(self.image_dataset.keys())

    def __getitem__(self, image_identifier):
        return self.load_case(image_identifier)

    @abstractmethod
    def load_case(self, identifier):
        pass

    @staticmethod
    @abstractmethod
    def save_case(
        data: np.ndarray,
        seg: np.ndarray,
        properties: dict,
        output_filename_truncated: str,
    ):
        pass

    @staticmethod
    @abstractmethod
    def get_identifiers(folder: str) -> List[str]:
        pass


class nnSSLDatasetBlosc2(nnSSLBaseDataset):

    def __init__(
        self,
        dataset_dir: str,
        collection: Collection,
        subject_identifiers: List[str] = None,
        iimg_filters: list[AbstractFilter] = None,
    ):
        """
        This is a dataset that allows loading data saved in blosc2 format.
        It will hold a
        """
        super().__init__(dataset_dir, collection, subject_identifiers, iimg_filters)
        blosc2.set_nthreads(1)

    def __getitem__(self, image_identifier):
        try:
            return self.load_case(
                self.dataset_dir, self.image_dataset, image_identifier
            )
        except RuntimeError as e:
            return self.__getitem__(choice(self.image_identifiers))

    @staticmethod
    def load_case(
        dataset_dir: str,
        image_dataset: dict[str, IndependentImage],
        image_identifier: str,
    ):
        dparams = {"nthreads": 1}
        img: IndependentImage
        img = image_dataset[image_identifier]
        output_img_path = img.get_output_path("image", ext=".b2nd")
        output_img_pkl_path = img.get_output_path("image", ext=".pkl")
        output_anat_mask_path = img.get_output_path("anat_mask", ext=".b2nd")
        output_anon_mask_path = img.get_output_path("anon_mask", ext=".b2nd")
        data_b2nd_file = join(dataset_dir, output_img_path)
        data = blosc2.open(
            urlpath=data_b2nd_file, mode="r", dparams=dparams, mmap_mode="r"
        )

        anon_b2nd_file = join(dataset_dir, output_anon_mask_path)
        if isfile(anon_b2nd_file):
            anon = blosc2.open(
                urlpath=anon_b2nd_file, mode="r", dparams=dparams, mmap_mode="r"
            )
        else:
            anon = None

        anat_b2nd_file = join(dataset_dir, output_anat_mask_path)
        if isfile(anat_b2nd_file):
            anat = blosc2.open(
                urlpath=anat_b2nd_file, mode="r", dparams=dparams, mmap_mode="r"
            )
        else:
            anat = None

        properties = load_pickle(join(dataset_dir, output_img_pkl_path))
        return data, anon, anat, properties

    @staticmethod
    def verify_file_exists(
        image_identifier: str,
        dataset_dir: str,
        image_dataset: dict[str, IndependentImage],
    ) -> tuple[bool, bool, bool]:
        img: IndependentImage
        img = image_dataset[image_identifier]
        output_img_path = img.get_output_path("image", ext=".b2nd")
        output_img_pkl_path = img.get_output_path("image", ext=".pkl")
        output_anat_mask_path = img.get_output_path("anat_mask", ext=".b2nd")
        output_anon_mask_path = img.get_output_path("anon_mask", ext=".b2nd")
        data_b2nd_file = join(dataset_dir, output_img_path)
        data_and_pkl_exists = isfile(data_b2nd_file) and isfile(
            join(dataset_dir, output_img_pkl_path)
        )

        anon_b2nd_file = join(dataset_dir, output_anon_mask_path)
        anon_exists = isfile(anon_b2nd_file)

        anat_b2nd_file = join(dataset_dir, output_anat_mask_path)
        anat_exists = isfile(anat_b2nd_file)

        return data_and_pkl_exists, anon_exists, anat_exists

    @staticmethod
    def save_case(
        data: np.ndarray,
        anon_mask: np.ndarray | None,
        anat_mask: np.ndarray | None,
        properties: dict,
        output_filename_truncated: str,
        anon_mask_filename_truncated: str,
        anat_mask_filename_truncated: str,
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
                urlpath=anon_mask_filename_truncated + ".b2nd",
                chunks=chunks_seg,
                blocks=blocks_seg,
                cparams=cparams,
                mmap_mode="w+",
            )

        if anat_mask is not None:
            blosc2.asarray(
                np.ascontiguousarray(anat_mask),
                urlpath=anat_mask_filename_truncated + ".b2nd",
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
        raise NotImplementedError(
            "This method is not supposed to be called for nnSSLDatasetBlosc2"
        )
        case_identifiers = [
            i[:-5]
            for i in os.listdir(folder)
            if i.endswith(".b2nd") and not i.endswith("_seg.b2nd")
        ]
        return case_identifiers

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
        block_size = np.array(
            (
                num_channels,
                *[2 ** (max(0, math.floor(math.log2(i / 2)))) for i in patch_size],
            )
        )

        # shrink the block size until it fits in L1
        estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel
        while estimated_nbytes_block > (
            l1_cache_size_per_core_in_bytes * safety_factor
        ):
            # pick largest deviation from patch_size that is not 1
            axis_order = np.argsort(block_size[1:] / patch_size)[::-1]
            idx = 0
            picked_axis = axis_order[idx]
            while (
                block_size[picked_axis + 1] == 1
                or block_size[picked_axis + 1] == image_size[picked_axis + 1]
            ):
                idx += 1
                picked_axis = axis_order[idx]
            # now reduce that axis to the next lowest power of 2
            block_size[picked_axis + 1] = 2 ** (
                max(0, math.floor(math.log2(block_size[picked_axis + 1] - 1)))
            )
            block_size[picked_axis + 1] = min(
                block_size[picked_axis + 1], image_size[picked_axis + 1]
            )
            estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel
            if all([i == j for i, j in zip(block_size, image_size)]):
                break

        # note: there is no use extending the chunk size to 3d when we have a 2d patch size! This would unnecessarily
        # load data into L3
        # now tile the blocks into chunks until we hit image_size or the l3 cache per core limit
        chunk_size = deepcopy(block_size)
        estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
        while estimated_nbytes_chunk < (
            l3_cache_size_per_core_in_bytes * safety_factor
        ):
            # find axis that deviates from block_size the most
            axis_order = np.argsort(chunk_size[1:] / block_size[1:])
            idx = 0
            picked_axis = axis_order[idx]
            while (
                chunk_size[picked_axis + 1] == image_size[picked_axis + 1]
                or patch_size[picked_axis] == 1
            ):
                idx += 1
                picked_axis = axis_order[idx]
            chunk_size[picked_axis + 1] += block_size[picked_axis + 1]
            chunk_size[picked_axis + 1] = min(
                chunk_size[picked_axis + 1], image_size[picked_axis + 1]
            )
            estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
            if patch_size[0] == 1:
                if all([i == j for i, j in zip(chunk_size[2:], image_size[2:])]):
                    break
            if all([i == j for i, j in zip(chunk_size, image_size)]):
                break
        # print(image_size, chunk_size, block_size)
        return tuple(block_size), tuple(chunk_size)


file_ending_dataset_mapping = {"b2nd": nnSSLDatasetBlosc2}


def infer_dataset_class(folder: str) -> Union[Type[nnSSLDatasetBlosc2]]:
    file_endings = set(
        [os.path.basename(i).split(".")[-1] for i in subfiles(folder, join=False)]
    )
    if "pkl" in file_endings:
        file_endings.remove("pkl")
    if "npy" in file_endings:
        file_endings.remove("npy")
    assert len(file_endings) == 1, (
        f"Found more than one file ending in the folder {folder}. "
        f"Unable to infer nnUNetDataset variant!"
    )
    return file_ending_dataset_mapping[list(file_endings)[0]]
