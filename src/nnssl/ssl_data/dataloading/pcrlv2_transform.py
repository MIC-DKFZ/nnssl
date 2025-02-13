from itertools import combinations
from math import prod, ceil
from random import choice, choices
from numpy.random import randint
from typing import TypeAlias
from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np
from skimage.transform import resize
from einops.einops import rearrange


BoundingBox3D: TypeAlias = tuple[int, int, int, int, int, int]
Shape3D: TypeAlias = tuple[int, int, int]

class PCRLv2Transform(AbstractTransform):

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
        self.global_patch_sizes = global_patch_sizes
        self.global_input_size = global_input_size
        self.local_patch_sizes = local_patch_sizes
        self.local_input_size = local_input_size
        self.num_locals = num_locals
        self.min_IoU = min_IoU

    def __call__(self, **data_dict):
        imgs = data_dict.get(self.data_key)
        if imgs is None:
            raise ValueError(f"No data found for key {self.data_key}")

        global_crops, local_crops = self.get_global_and_local_crops(imgs)
        # global_crops: [B,          2, C, X_global_input_size, Y_global_input_size, Z_global_input_size]
        # local_crops:  [B, num_locals, C, X_local_input_size,  Y_local_input_size,  Z_local_input_size]
        B = imgs.shape[0]


        return new_data_dict

    def get_global_and_local_crops(self, imgs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        batch_size, _, X, Y, Z = imgs.shape
        all_global_crop_pairs, all_local_crops = [], []
        for i in range(batch_size):
            image = imgs[i]
            g_patch_size_A, g_patch_size_B = choices(self.global_patch_sizes, k=2)
            big_bbox = self.get_big_bbox((X, Y, Z), g_patch_size_A, g_patch_size_B)    # [x_start, y_start, z_start, xs, ys, zs]
            g_bbox_A, g_bbox_B = self.get_global_bboxes(g_patch_size_A, g_patch_size_B, big_bbox)

            big_crop = self.get_crop(image, big_bbox)
            g_crop_A, g_crop_B = self.get_crop(big_crop, g_bbox_A), self.get_crop(big_crop, g_bbox_B)
            g_crop_A, g_crop_B = (resize(g_crop_A, self.global_input_size, preserve_range=True),
                                  resize(g_crop_B, self.global_input_size, preserve_range=True))

            min_bbox = self.get_min_bbox(g_bbox_A, g_bbox_B)
            min_crop = self.get_crop(image, min_bbox)
            local_crops = []
            for _ in range(self.num_locals):
                local_patch_size = choice(self.local_patch_sizes)
                l_bbox = self.get_rand_inner_bbox(min_bbox[3:], local_patch_size)
                l_crop = resize(self.get_crop(min_crop, l_bbox), self.local_input_size, preserve_range=True)
                local_crops.append(l_crop)

            all_global_crop_pairs.append(np.stack((g_crop_A, g_crop_B), axis=1))    # list of [1, 2, C, X, Y, Z]
            all_local_crops.append(np.stack(local_crops, axis=1))   # list of [1, 6, C, X, Y, Z]

        all_global_crop_pairs = np.concat(all_global_crop_pairs, axis=0)    # [B, 2, C, X, Y, Z]
        all_local_crops = np.concat(all_local_crops, axis=0)                # [B, 6, C, X, Y, Z]

    def get_global_bboxes(self, g_patch_size_A: Shape3D, g_patch_size_B: Shape3D, big_bbox: Shape3D):
        pass

    @staticmethod
    def calculate_IoU(bbox_1: BoundingBox3D, bbox_2: BoundingBox3D) -> int:
        overlaps_per_axis = [
            max(0, min(bbox_1[0 + i] + bbox_1[3 + i], bbox_2[0 + i] + bbox_2[3 + i]) - max(bbox_1[0 + i], bbox_2[0 + i]))
            for i in range(3)]
        return prod(overlaps_per_axis)

    @staticmethod
    def get_big_bbox(img_shape: Shape3D, g_patch_size_A: Shape3D, g_patch_size_B: Shape3D,
                     min_IoU: float) -> Shape3D:
        """
        The original implementation gets two global views with an IoU restriction by randomly sampling two global
        global views repeatedly from the image until the IoU threshold is reached. While this is not optimal, the smallest
        possible bbox from which you (1) can sample the two global views randomly, (2) still be above the IoU
        threshold and (3) still have all possible combinations is not a rectangular cuboid, making it difficult to
        calculate it.
        This function tries to provide a bbox where the possibility of not reaching the IoU threshold is minimized,
        so that if you randomly sample two global views from this bbox, the number of  iterations it takes to meet the
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
            min_side_intersection = ceil( (volume_A+volume_B)*min_IoU / ((1+min_IoU)*max_areas[i]) )
            big_bbox_shape.append(side_A + side_B - min_side_intersection)

        big_bbox_starts = [randint(0, img_shape[i] - big_bbox_shape[i] + 1) for i in range(3)]

        return tuple(big_bbox_starts + big_bbox_shape)

    @staticmethod
    def get_crop(image: np.ndarray, bbox: BoundingBox3D):
        x_start, y_start, z_start, xs, ys, zs = bbox
        return image[:, x_start:x_start+xs, y_start:y_start+ys, z_start:z_start+zs]


    @staticmethod
    def get_rand_inner_bbox(big_bbox_shape: Shape3D, target_bbox_shape: Shape3D) -> BoundingBox3D:
        pass





if __name__ == "__main__":
    # bbox_1 = (2, 4, 3, 4, 4, 6)
    # bbox_2 = (1, 3, 2, 2, 4, 2)
    #
    # print(PCRLv2Transform.calculate_IoU(bbox_1, bbox_2))
    pass













