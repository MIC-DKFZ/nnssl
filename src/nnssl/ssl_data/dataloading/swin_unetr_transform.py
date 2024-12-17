from batchgenerators.transforms.abstract_transforms import AbstractTransform
from random import randint
import numpy as np


def patch_rand_drop(img: np.ndarray, img_rep: None | np.ndarray = None, max_drop=0.3, max_block_sz=0.25, tolr=0.05):
    c, h, w, z = img.shape
    n_drop_pix = np.random.uniform(0, max_drop) * h * w * z
    mx_blk_height = int(h * max_block_sz)
    mx_blk_width = int(w * max_block_sz)
    mx_blk_slices = int(z * max_block_sz)
    tolr = (int(tolr * h), int(tolr * w), int(tolr * z))
    total_pix = 0
    while total_pix < n_drop_pix:
        rnd_r = randint(0, h - tolr[0])
        rnd_c = randint(0, w - tolr[1])
        rnd_s = randint(0, z - tolr[2])
        rnd_h = min(randint(tolr[0], mx_blk_height) + rnd_r, h)
        rnd_w = min(randint(tolr[1], mx_blk_width) + rnd_c, w)
        rnd_z = min(randint(tolr[2], mx_blk_slices) + rnd_s, z)
        if img_rep is None:
            noisy_patch = np.random.normal(loc=0.0, scale=1.0,
                                                     size=(c, rnd_h - rnd_r, rnd_w - rnd_c, rnd_z - rnd_s))
            noisy_patch = (noisy_patch - np.min(noisy_patch)) / (
                    np.max(noisy_patch) - np.min(noisy_patch)
            )
            img[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z] = noisy_patch
        else:
            img[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z] = img_rep[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z]
        total_pix = total_pix + (rnd_h - rnd_r) * (rnd_w - rnd_c) * (rnd_z - rnd_s)
    return img


def inner_cutout_rand(imgs: np.ndarray):
    """
    Section 5.2 and Appendix E in the paper mention a ROI dropping rate of 30%. However, the official
    implementation first replaces 30% with random noise and optionally replaces another 30% with corresponding subvolumes
    of a different random case within the batch, leading to a maximam potential drop rate of 60% (if non-overlapping
    drop regions).
    https://github.com/Project-MONAI/research-contributions/blob/207cad9b2f15c958fcb5d9594ddaeca61f8f3dd6/SwinUNETR/Pretrain/utils/ops.py#L67
    """
    img_n = imgs.shape[0]
    imgs_aug = imgs.copy()
    for i in range(img_n):
        imgs_aug[i] = patch_rand_drop(imgs_aug[i])
        idx_rnd = randint(0, img_n-1)
        if idx_rnd != i:
            imgs_aug[i] = patch_rand_drop(imgs_aug[i], imgs_aug[idx_rnd])
    return imgs_aug


def rot_rand(imgs: np.ndarray):
    img_n = imgs.shape[0]
    imgs_aug = imgs.copy()
    rotations = np.zeros(img_n, dtype=np.int64)
    for i in range(img_n):
        x = imgs[i]
        rotation = np.random.randint(0, 4)
        if rotation == 0:
            pass
        elif rotation == 1:
            x = np.rot90(x, 1, (2, 3))
        elif rotation == 2:
            x = np.rot90(x, 2, (2, 3))
        elif rotation == 3:
            x = np.rot90(x, 3, (2, 3))
        imgs_aug[i] = x
        rotations[i] = rotation
    return imgs_aug, rotations


class SwinUNETRTransform(AbstractTransform):
    def __init__(self, data_key="data"):
        self.data_key = data_key

    def __call__(self, **data_dict):
        imgs = data_dict.get(self.data_key)
        if imgs is None:
            raise ValueError(f"No data found for key {self.data_key}")

        imgs1_rotated, rotations1 = rot_rand(imgs)
        imgs1_rotated_cutout = inner_cutout_rand(imgs1_rotated)
        imgs2_rotated, rotations2 = rot_rand(imgs)
        imgs2_rotated_cutout = inner_cutout_rand(imgs2_rotated)

        new_data_dict = {
            "imgs_rotated": (imgs1_rotated, imgs2_rotated),
            "rotations": (rotations1, rotations2),
            "imgs_rotated_cutout": (imgs1_rotated_cutout, imgs2_rotated_cutout)
        }

        return new_data_dict



# if __name__ == "__main__":
#     tr = SwinUNETRTransform()
#     x = np.random.normal(size=(2, 1, 96, 96, 96))
#     tr(data=x)

























