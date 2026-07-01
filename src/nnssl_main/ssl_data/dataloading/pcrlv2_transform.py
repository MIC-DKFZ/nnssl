import gc
from itertools import combinations, combinations_with_replacement
from math import prod, ceil
from random import choice

import torch
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform, SpatialTransform
from numpy.random import randint
from typing import TypeAlias
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
import numpy as np
import torch.nn.functional as F
import torchio

BoundingBox3D: TypeAlias = tuple[int, int, int, int, int, int]  # x_start, y_start, z_start, xs, ys, zs
Shape3D: TypeAlias = tuple[int, int, int]   # x_size, y_size, z_size


class RandomSwap(AbstractTransform):

    def __init__(self, patch_size: tuple[int, int, int], num_swaps: int, data_key="data", label_key="seg"):
        self.data_key = data_key
        self.label_key = label_key
        self.d, self.h, self.w = patch_size
        self.num_swaps = num_swaps

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)

        patch_d, patch_h, patch_w = self.d, self.h, self.w
        _, _, depth, height, width = data.shape
        aug_data = data.copy()

        for b in range(len(data)):
            for _ in range(self.num_swaps):
                d1, h1, w1 = np.random.randint(0, depth - patch_d), np.random.randint(0, height - patch_h), np.random.randint(0, width - patch_w)
                d2, h2, w2 = np.random.randint(0, depth - patch_d), np.random.randint(0, height - patch_h), np.random.randint(0, width - patch_w)

                patch1 = aug_data[b, :, d1:d1 + patch_d, h1:h1 + patch_h, w1:w1 + patch_w].copy()
                patch2 = aug_data[b, :, d2:d2 + patch_d, h2:h2 + patch_h, w2:w2 + patch_w].copy()

                aug_data[b, :, d1:d1 + patch_d, h1:h1 + patch_h, w1:w1 + patch_w] = patch2
                aug_data[b, :, d2:d2 + patch_d, h2:h2 + patch_h, w2:w2 + patch_w] = patch1

        data_dict[self.data_key] = aug_data
        return data_dict


class PCLRv2Transform(AbstractTransform):

    def __init__(
            self,
            global_patch_sizes: tuple[Shape3D],
            global_input_size: Shape3D,
            local_patch_sizes: tuple[Shape3D],
            local_input_size: Shape3D,
            num_locals: int,
            min_IoU: float,
            data_key="data"
        ):
        self.data_key = data_key
        self.global_input_size = global_input_size
        self.local_patch_sizes = local_patch_sizes
        self.local_input_size = local_input_size
        self.num_locals = num_locals
        self.min_IoU = min_IoU

        # Make sure that two global patch sizes can reach the IoU threshold given a maximum overlap
        self.global_patch_size_pairs = []
        for p1, p2 in combinations_with_replacement(global_patch_sizes, r=2):
            v1, v2 = prod(p1), prod(p2)
            max_overlap = prod(min(p1[i], p2[i]) for i in range(3))
            if max_overlap / (v1 + v2 - max_overlap) >= self.min_IoU:
                self.global_patch_size_pairs.append((p1, p2))

        # self.spatial_transforms = torchio.transforms.Compose([
        #     torchio.transforms.RandomFlip(),
            torchio.transforms.RandomAffine(),
        # ])
        # self.local_transforms = torchio.transforms.Compose([
        #     torchio.transforms.RandomBlur(),
        #     torchio.transforms.RandomNoise(),
        #     torchio.transforms.RandomGamma(),
        #     # torchio.transforms.ZNormalization()       # we already do z-norm in our pre-processing
        # ])
        # self.global_transforms = torchio.transforms.Compose([
        #     torchio.transforms.RandomBlur(),
        #     torchio.transforms.RandomNoise(),
        #     torchio.transforms.RandomGamma(),
        #     torchio.transforms.RandomSwap(patch_size=(12, 12, 12)),
        #     torchio.transforms.RandomSwap(patch_size=(8, 4, 4)),
        #     # torchio.transforms.ZNormalization()
        # ])

        self.global_spatial_transforms = Compose([
            MirrorTransform(axes=(0, 1, 2)),
            SpatialTransform(patch_size=global_input_size,
                             angle_x=(-np.pi/18, np.pi/18),
                             angle_y=(-np.pi/18, np.pi/18),
                             angle_z=(-np.pi/18, np.pi/18),
                             do_elastic_deform=False,
                             scale=(0.9, 1.1),
                             random_crop=False)
        ])

        self.global_visual_transforms = Compose([
            GaussianNoiseTransform(noise_variance=(0, 0.25)),
            GaussianBlurTransform(blur_sigma=(0, 2)),
            GammaTransform((0.75, 1.35))
        ])

        self.local_transforms = Compose([
            MirrorTransform(axes=(0, 1, 2)),
            SpatialTransform(patch_size=local_input_size,
                             angle_x=(-np.pi/18, np.pi/18),
                             angle_y=(-np.pi/18, np.pi/18),
                             angle_z=(-np.pi/18, np.pi/18),
                             do_elastic_deform=False,
                             scale=(0.9, 1.1),
                             random_crop=False),
            GaussianNoiseTransform(noise_variance=(0, 0.25)),
            GaussianBlurTransform(blur_sigma=(0, 2)),
            GammaTransform((0.75, 1.35))
            # RandomSwap(patch_size=(12, 12, 12), num_swaps=50)
        ])

    # def apply_global_transforms(self, global_crops: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    #     sp_global_crops = []
    #     aug_global_crops = []
    #     for i in range(global_crops.shape[0]):
    #         sp_global_crops.append(self.spatial_transforms(global_crops[i]))
    #         aug_global_crops.append(self.global_transforms(sp_global_crops[i]))
    #     return np.stack(sp_global_crops), np.stack(aug_global_crops)
    #
    #
    # def apply_local_transforms(self, local_crops: torch.Tensor) -> torch.Tensor:
    #     aug_local_crops = []
    #     for i in range(local_crops.shape[0]):
    #         aug_local_crops.append(self.local_transforms(self.spatial_transforms(local_crops[i])))
    #     return np.stack(aug_local_crops)


    def __call__(self, **data_dict):
        imgs = data_dict.get(self.data_key)
        if imgs is None:
            raise ValueError(f"No data found for key {self.data_key}")

        global_crops_A, global_crops_B, local_crops = self.get_global_and_local_crops(imgs)

        # global_crops_A, aug_global_crops_A = self.apply_global_transforms(global_crops_A.numpy())
        # global_crops_B, aug_global_crops_B = self.apply_global_transforms(global_crops_B.numpy())
        # aug_local_crops = self.apply_local_transforms(local_crops.numpy())

        global_crops_A = self.global_spatial_transforms(data=global_crops_A)["data"]
        aug_global_crops_A = self.global_visual_transforms(data=global_crops_A)["data"]

        global_crops_B = self.global_spatial_transforms(data=global_crops_B)["data"]
        aug_global_crops_B = self.global_visual_transforms(data=global_crops_B)["data"]

        aug_local_crops = self.local_transforms(data=local_crops)["data"]

        #DEBUG
        # aug_global_crops_A = global_crops_A
        # aug_global_crops_B = global_crops_B
        # aug_local_crops = local_crops

        new_data_dict = {
            "aug_global_crops_A": aug_global_crops_A,
            "global_crops_A": global_crops_A,
            "aug_global_crops_B": aug_global_crops_B,
            "aug_local_crops": aug_local_crops,
        }
        return new_data_dict

    def get_global_and_local_crops(self, imgs: np.ndarray):
        batch_size, N, X, Y, Z = imgs.shape
        global_crops_A, global_crops_B, all_local_crops = [], [], []
        for i in range(batch_size):
            image = imgs[i]
            g_patch_size_A, g_patch_size_B = choice(self.global_patch_size_pairs)
            big_bbox = self.get_rand_big_bbox((X, Y, Z), g_patch_size_A, g_patch_size_B, self.min_IoU)   # [x_start, y_start, z_start, x_end, y_end, z_end]
            g_bbox_A, g_bbox_B = self.get_global_bboxes(g_patch_size_A, g_patch_size_B, big_bbox)

            g_crop_A, g_crop_B = self.get_crop(image, g_bbox_A), self.get_crop(image, g_bbox_B)

            # The original codebase uses skimage.transform.resize, which is very slow, that's also (partly) why they
            # moved their augmentations to the data preprocessing step, resulting in limited augmentations.
            g_crop_A, g_crop_B = F.interpolate(torch.from_numpy(g_crop_A)[None, ...], self.global_input_size).numpy(), \
                                 F.interpolate(torch.from_numpy(g_crop_B)[None, ...], self.global_input_size).numpy()

            # get the minimum sized bounding box that holds both global bboxes
            min_bbox = self.get_min_bbox(g_bbox_A, g_bbox_B)

            local_crops = []
            for _ in range(self.num_locals):
                local_patch_size = choice(self.local_patch_sizes)
                l_bbox = self.get_rand_inner_bbox(min_bbox, local_patch_size)
                l_crop = self.get_crop(image, l_bbox)
                l_crop = F.interpolate(torch.from_numpy(l_crop)[None, ...], self.local_input_size).numpy()
                local_crops.append(l_crop)

            global_crops_A.append(g_crop_A)
            global_crops_B.append(g_crop_B)
            all_local_crops.extend(local_crops)

        global_crops_A = np.concat(global_crops_A)                # [B, C, X, Y, Z]
        global_crops_B = np.concat(global_crops_B)                # [B, C, X, Y, Z]
        all_local_crops = np.concat(all_local_crops)              # [B*num_locals, C, X, Y, Z]

        return global_crops_A, global_crops_B, all_local_crops

    def get_global_bboxes(self, g_patch_size_A: Shape3D, g_patch_size_B: Shape3D, big_bbox: BoundingBox3D) -> tuple[
        BoundingBox3D, BoundingBox3D]:
        g_bbox_A = self.get_rand_inner_bbox(big_bbox, g_patch_size_A)
        # tries = 0
        while True:
            g_bbox_B = self.get_rand_inner_bbox(big_bbox, g_patch_size_B)
            # tries += 1
            if self.calculate_IoU(g_bbox_A, g_bbox_B) >= self.min_IoU:
                # print(tries)
                break
        return g_bbox_A, g_bbox_B


    def get_min_bbox(self, bbox_A: BoundingBox3D, bbox_B: BoundingBox3D) -> BoundingBox3D:
        min_bbox_starts = [min(bbox_A[i], bbox_B[i]) for i in range(3)]
        min_bbox_ends = [max(bbox_A[i] + bbox_A[3+i], bbox_B[i] + bbox_B[3+i]) for i in range(3)]
        min_bbox_shape = [min_bbox_ends[i] - min_bbox_starts[i] for i in range(3)]
        return BoundingBox3D(min_bbox_starts + min_bbox_shape)


    @staticmethod
    def calculate_IoU(bbox_1: BoundingBox3D, bbox_2: BoundingBox3D) -> float:
        overlaps_per_axis = [
            max(0, min(bbox_1[i] + bbox_1[3 + i], bbox_2[i] + bbox_2[3 + i]) - max(bbox_1[0 + i], bbox_2[0 + i]))
            for i in range(3)
        ]
        overlapping_volume = prod(overlaps_per_axis)
        v1, v2 = prod(bbox_1[3:]), prod(bbox_2[3:])
        return overlapping_volume / (v1 + v2 - overlapping_volume)


    @staticmethod
    def get_rand_big_bbox(img_shape: Shape3D, g_patch_size_A: Shape3D, g_patch_size_B: Shape3D,
                          min_IoU: float) -> BoundingBox3D:
        """
        The original implementation gets two global views with an IoU restriction by randomly sampling two global
        global views repeatedly from the image until the IoU threshold is reached. While this is not optimal, the smallest
        possible bbox from which you (1) can sample the two global views randomly, (2) still be above the IoU
        threshold and (3) still have all possible combinations is not a rectangular cuboid, making it difficult to
        calculate it.
        This function tries to provide a bbox where the possibility of not reaching the IoU threshold is minimized,
        so that if you randomly sample two global views from this bbox, the number of iterations it takes to meet the
        IoU restriction is minimized as well.
        """
        big_bbox_shape = []
        max_overlaps = [min(A, B) for A, B in zip(g_patch_size_A, g_patch_size_B)]
        max_areas = [side_1 * side_2 for side_1, side_2 in reversed(list(combinations(max_overlaps, r=2)))]
        volume_A, volume_B = prod(g_patch_size_A), prod(g_patch_size_B)

        # for each axis, calculate the minimum intersection of A and B, so that if the overlapping area of the
        # other two axis is maximal, the resulting IoU is above 'min_IoU'
        for i in range(3):
            side_A, side_B = g_patch_size_A[i], g_patch_size_B[i]

            # Q: How do we get the minimum intersection per axis/side, so that we are still over the IoU threshold
            #    given a maximal overlap area of the other two axis?
            # -> Start from this equation:
            #   ((min_side_intersect * max_area) / (V1 + V2 - min_side_intersect * max_area) > min_IoU
            # -> solve for min_side_intersect, then we get the following:
            min_side_intersection = ceil( (volume_A+volume_B)*min_IoU / ((1+min_IoU)*max_areas[i]) )
            big_bbox_shape.append(min(img_shape[i], side_A + side_B - min_side_intersection))

        big_bbox_starts = [randint(0, img_shape[i] - big_bbox_shape[i] + 1) for i in range(3)]
        return BoundingBox3D(big_bbox_starts + big_bbox_shape)

    @staticmethod
    def get_crop(image: np.ndarray, bbox: BoundingBox3D):
        x_start, y_start, z_start, xs, ys, zs = bbox
        return image[:, x_start:x_start+xs, y_start:y_start+ys, z_start:z_start+zs]


    @staticmethod
    def get_rand_inner_bbox(bbox: BoundingBox3D, inner_bbox_shape: Shape3D) -> BoundingBox3D:
        inner_bbox_start = tuple([bbox[i] + randint(0, bbox[3+i] - inner_bbox_shape[i]) for i in range(3)])
        return BoundingBox3D(inner_bbox_start + inner_bbox_shape)



if __name__ == "__main__":
    # bbox_1 = (2, 4, 3, 4, 4, 6)
    # bbox_2 = (1, 3, 2, 2, 4, 2)
    # print(PCRLv2Transform.calculate_IoU(bbox_1, bbox_2))

    trafo = PCLRv2Transform(
        global_patch_sizes = ((96, 96, 96), (128, 128, 96), (128, 128, 128), (160, 160, 128)),
        global_input_size = (128, 128, 128),
        local_patch_sizes = ((32, 32, 32), (64, 64, 32), (64, 64, 64)),
        local_input_size = (64, 64, 64),
        num_locals = 6,
        min_IoU = 0.3
    )

    # _ = trafo.get_rand_big_bbox((160, 160, 160), (96, 96, 96), (96, 96, 96), 0.3)

    bbox = (20, 31, 127, 112, 112, 64)

    import time

    with torch.no_grad():
        start = time.time()
        for i in range(4):
            imgs = np.zeros((8, 1, 180, 180, 180))
            _ = trafo(data=imgs)
        elapsed = time.time() - start
        print(f"Time per iteration: {elapsed/4:.3f}s")













