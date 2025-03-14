from typing import Union, List, Tuple
from unittest.mock import patch

import numpy as np
from nnssl.ssl_data.dataloading.base_data_loader import nnsslDataLoaderBase
from nnssl.data.dataloading.dataset import nnSSLDatasetBlosc2


class nnsslDataLoader3D(nnsslDataLoaderBase):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        anon_all = np.zeros(self.data_shape, dtype=np.uint8)
        # anat_all = np.zeros(self.data_shape, dtype=np.uint8)
        case_properties = []

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)

            data, anon, anat, properties = self._data[i]
            if anon is None:
                anon = np.zeros(data.shape, dtype=np.uint8)
            # if anat is None:
            #     anat = np.zeros(data.shape, dtype=np.uint8)
            case_properties.append(properties)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape)

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple(
                [slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)]
            )
            data = data[this_slice]
            anon = anon[this_slice]
            # anat = anat[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            data_all[j] = np.pad(data, ((0, 0), *padding), "constant", constant_values=0)
            anon_all[j] = np.pad(anon, ((0, 0), *padding), "constant", constant_values=0)
            # anat_all[j] = np.pad(anat, ((0, 0), *padding), "constant", constant_values=0)

        return {
            "data": data_all,
            "seg": anon_all,
            # "anat": anat_all,
            "properties": case_properties,
            "keys": selected_keys,
        }


class nnsslAnatDataLoader3D(nnsslDataLoaderBase):

    def __init__(
        self,
        data: nnSSLDatasetBlosc2,
        batch_size: int,
        patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
        final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
        sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
        pad_sides: Union[List[int], Tuple[int, ...], np.ndarray] = None,
        oversample_foreground_percent: float = 0.33
    ):
        super().__init__(data, batch_size, patch_size, final_patch_size, sampling_probabilities, pad_sides)
        self.oversample_foreground_percent = oversample_foreground_percent

    def _probabilistic_oversampling(self) -> bool:
        return np.random.uniform() <= self.oversample_foreground_percent

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = []
        anon_all = []
        case_properties = []

        for i in selected_keys:
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)

            data, anon, anat, properties = self._data[i]
            if anon is None:
                anon = np.zeros(data.shape, dtype=np.uint8)
            if anat is None:
                anat = np.zeros(data.shape, dtype=np.uint8)
            case_properties.append(properties)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            force_fg = self._probabilistic_oversampling()
            bbox_lbs, bbox_ubs = self.get_bbox(shape, anat if force_fg else None)

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple(
                [slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)]
            )
            data = data[this_slice]
            anon = anon[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            data_all.append(np.pad(data, ((0, 0), *padding), "constant", constant_values=0))
            anon_all.append(np.pad(anon, ((0, 0), *padding), "constant", constant_values=0))

        data_all = np.stack(data_all, axis=0)
        anon_all = np.stack(anon_all, axis=0)

        return {
            "data": data_all,
            "seg": anon_all,
            "properties": case_properties,
            "keys": selected_keys,
        }

    def get_bbox(
            self,
            data_shape: np.ndarray,
            anat: np.ndarray | None
    ):
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        for d in range(dim):
            # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            # always
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
        # define what the upper and lower bound can be to then sample form them with np.random.randint
        lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]

        if anat is not None:
            foreground_voxels = np.argwhere(1 - anat[0, :])
            if len(foreground_voxels) > 0:
                selected_voxel = tuple(foreground_voxels[np.random.choice(len(foreground_voxels))])
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                bbox_lbs = [max(lbs[i], selected_voxel[i] - self.patch_size[i] // 2) for i in range(dim)]
            else:
                anat = None

        if anat is None:
            # If the image does not contain any foreground classes, we fall back to random cropping
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]

        return bbox_lbs, bbox_ubs


class nnsslCenterCropDataLoader3D(nnsslDataLoaderBase):

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = []
        anon_all = []
        # anat_all = []
        case_properties = []

        for i in selected_keys:
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)

            data, anon, anat, properties = self._data[i]
            if anon is None:
                anon = np.zeros(data.shape, dtype=np.uint8)
            # if anat is None:
            #     anat = np.zeros(data.shape, dtype=np.uint8)
            case_properties.append(properties)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape)  # Returns the Center Crop.

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple(
                [slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)]
            )
            data = data[this_slice]
            anon = anon[this_slice]
            # anat = anat[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            data_all.append(np.pad(data, ((0, 0), *padding), "constant", constant_values=0))
            anon_all.append(np.pad(anon, ((0, 0), *padding), "constant", constant_values=0))
            # anat_all.append(np.pad(anat, ((0, 0), *padding), "constant", constant_values=0))

        data_all = np.stack(data_all, axis=0)
        anon_all = np.stack(anon_all, axis=0)
        # anat_all = np.stack(anat_all, axis=0)

        return {
            "data": data_all,
            "seg": anon_all,
            # "anat_mask": anat_all,
            "properties": case_properties,
            "keys": selected_keys,
        }

    def get_bbox(
        self,
        data_shape: np.ndarray,
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
        bbox_lbs = [int((lbs[i] + ubs[i]) / 2) for i in range(dim)]  # Always take the center (after padding)
        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]

        return bbox_lbs, bbox_ubs


class nnsslIndexableCenterCropDataLoader3D(nnsslDataLoaderBase):

    def __init__(
        self,
        data: nnSSLDatasetBlosc2,
        batch_size: int,
        patch_size: list[int]| tuple[int, ...]| np.ndarray,
        final_patch_size: list[int]| tuple[int, ...]| np.ndarray,
        sampling_probabilities: list[int]| tuple[int, ...]| np.ndarray = None,
        pad_sides: list[int]| tuple[int, ...]| np.ndarray = None,
        max_samples: int | None = None
    ):
        data.image_identifiers = data.image_identifiers[:max_samples]
        data.image_dataset = {k: v for k, v in data.image_dataset.items() if k in data.image_identifiers}
        super().__init__(data, batch_size, patch_size, final_patch_size, sampling_probabilities, pad_sides)

    def generate_train_batch(self, index):
        offset = index * self.batch_size

        case_properties = []
        selected_keys = []
        data_all = []
        anon_all = []
        # anat_all = []

        for i in range(self.batch_size):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            try:
                selected_key = self.indices[offset + i]
            except IndexError:
                continue
            selected_keys.append(selected_key)

            data, anon, anat, properties = self._data[selected_key]
            if anon is None:
                anon = np.zeros(data.shape, dtype=np.uint8)
            # if anat is None:
            #     anat = np.zeros(data.shape, dtype=np.uint8)
            case_properties.append(properties)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape)  # Returns the Center Crop.

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple(
                [slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)]
            )
            data = data[this_slice]
            anon = anon[this_slice]
            # anat = anat[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            data_all.append(np.pad(data, ((0, 0), *padding), "constant", constant_values=0))
            anon_all.append(np.pad(anon, ((0, 0), *padding), "constant", constant_values=0))
            # anat_all.append(np.pad(anat, ((0, 0), *padding), "constant", constant_values=0))

        data_all = np.stack(data_all, axis=0)
        anon_all = np.stack(anon_all, axis=0)
        # anat_all = np.stack(anat_all, axis=0)

        return {
            "data": data_all,
            "seg": anon_all,
            # "anat_mask": anat_all,
            "properties": case_properties,
            "keys": selected_keys,
        }

    def get_bbox(
        self,
        data_shape: np.ndarray,
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
        bbox_lbs = [int((lbs[i] + ubs[i]) / 2) for i in range(dim)]  # Always take the center (after padding)
        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]

        return bbox_lbs, bbox_ubs

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        return self.generate_train_batch(index)


if __name__ == "__main__":
    print("Not intended as entry point!")
