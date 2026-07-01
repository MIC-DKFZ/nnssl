# Code taken from original Repository https://github.com/MrGiovanni/ModelsGenesis

from dataclasses import dataclass
import os
import random
import copy
import imageio
import string
import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform

try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb


@dataclass
class ModelGenesisConfig:
    # model = "Unet3D"
    # suffix = "genesis_chest_ct"
    # exp_name = model + "-" + suffix

    # data -- Don't need these infos
    # data = "/mnt/dataset/shared/zongwei/LUNA16/Self_Learning_Cubes"
    # train_fold = [0, 1, 2, 3, 4]
    # valid_fold = [5, 6]
    # test_fold = [7, 8, 9]
    # hu_min = -1000.0
    # hu_max = 1000.0
    # scale = 32
    # input_rows = 64
    # input_cols = 64
    # input_deps = 32
    # nb_class = 1

    # model pre-training
    # verbose = 1
    # weights = None
    # batch_size = 6
    # optimizer = "sgd"
    # workers = 10
    # max_queue_size = workers * 4
    # save_samples = "png"
    # nb_epoch = 10000
    # patience = 50
    # lr = 1

    # image deformation
    nonlinear_rate = 0.9
    paint_rate = 0.9
    outpaint_rate = 0.8
    inpaint_rate = 1.0 - outpaint_rate
    local_rate = 0.5
    flip_rate = 0.4

    # def display(self):
    #     """Display Configuration values."""
    #     print("\nConfigurations:")
    #     for a in dir(self):
    #         if not a.startswith("__") and not callable(getattr(self, a)):
    #             print("{:30} {}".format(a, getattr(self, a)))
    #     print("\n")


def bernstein_poly(i, n, t):
    """
    The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i


def bezier_curve(points, nTimes=1000):
    """
    Given a set of control points, return the
    bezier curve defined by the control points.

    Control points should be a list of lists, or list of tuples
    such as [ [1,1],
              [2,3],
              [4,5], ..[Xn, Yn] ]
     nTimes is the number of time steps, defaults to 1000

     See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def data_augmentation(x, y, seg, prob=0.5):
    # augmentation by flipping
    cnt = 3
    while random.random() < prob and cnt > 0:
        degree = random.choice([0, 1, 2])
        x = np.flip(x, axis=degree)
        y = np.flip(y, axis=degree)
        seg = np.flip(seg, axis=degree)
        cnt = cnt - 1

    return x, y, seg


def nonlinear_transformation(x, prob=0.5):
    if random.random() >= prob:
        return x
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x


def local_pixel_shuffling(x, prob=0.5):
    if random.random() >= prob:
        return x
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    _, img_rows, img_cols, img_deps = x.shape
    num_block = 10000
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows // 10)
        block_noise_size_y = random.randint(1, img_cols // 10)
        block_noise_size_z = random.randint(1, img_deps // 10)
        noise_x = random.randint(0, img_rows - block_noise_size_x)
        noise_y = random.randint(0, img_cols - block_noise_size_y)
        noise_z = random.randint(0, img_deps - block_noise_size_z)
        window = orig_image[
            0,
            noise_x : noise_x + block_noise_size_x,
            noise_y : noise_y + block_noise_size_y,
            noise_z : noise_z + block_noise_size_z,
        ]
        window = window.flatten()
        np.random.shuffle(window)
        window = window.reshape((block_noise_size_x, block_noise_size_y, block_noise_size_z))
        image_temp[
            0,
            noise_x : noise_x + block_noise_size_x,
            noise_y : noise_y + block_noise_size_y,
            noise_z : noise_z + block_noise_size_z,
        ] = window
    local_shuffling_x = image_temp

    return local_shuffling_x


def image_in_painting(x):
    _, img_rows, img_cols, img_deps = x.shape
    cnt = 5
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = random.randint(img_rows // 6, img_rows // 3)
        block_noise_size_y = random.randint(img_cols // 6, img_cols // 3)
        block_noise_size_z = random.randint(img_deps // 6, img_deps // 3)
        noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
        noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
        noise_z = random.randint(3, img_deps - block_noise_size_z - 3)
        x[
            :,
            noise_x : noise_x + block_noise_size_x,
            noise_y : noise_y + block_noise_size_y,
            noise_z : noise_z + block_noise_size_z,
        ] = (
            np.random.rand(
                block_noise_size_x,
                block_noise_size_y,
                block_noise_size_z,
            )
            * 1.0
        )
        cnt -= 1
    return x


def image_out_painting(x):
    _, img_rows, img_cols, img_deps = x.shape
    image_temp = copy.deepcopy(x)
    x = (
        np.random.rand(
            x.shape[0],
            x.shape[1],
            x.shape[2],
            x.shape[3],
        )
        * 1.0
    )
    block_noise_size_x = img_rows - random.randint(3 * img_rows // 7, 4 * img_rows // 7)
    block_noise_size_y = img_cols - random.randint(3 * img_cols // 7, 4 * img_cols // 7)
    block_noise_size_z = img_deps - random.randint(3 * img_deps // 7, 4 * img_deps // 7)
    noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
    noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
    noise_z = random.randint(3, img_deps - block_noise_size_z - 3)
    x[
        :,
        noise_x : noise_x + block_noise_size_x,
        noise_y : noise_y + block_noise_size_y,
        noise_z : noise_z + block_noise_size_z,
    ] = image_temp[
        :,
        noise_x : noise_x + block_noise_size_x,
        noise_y : noise_y + block_noise_size_y,
        noise_z : noise_z + block_noise_size_z,
    ]
    cnt = 4
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = img_rows - random.randint(3 * img_rows // 7, 4 * img_rows // 7)
        block_noise_size_y = img_cols - random.randint(3 * img_cols // 7, 4 * img_cols // 7)
        block_noise_size_z = img_deps - random.randint(3 * img_deps // 7, 4 * img_deps // 7)
        noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
        noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
        noise_z = random.randint(3, img_deps - block_noise_size_z - 3)
        x[
            :,
            noise_x : noise_x + block_noise_size_x,
            noise_y : noise_y + block_noise_size_y,
            noise_z : noise_z + block_noise_size_z,
        ] = image_temp[
            :,
            noise_x : noise_x + block_noise_size_x,
            noise_y : noise_y + block_noise_size_y,
            noise_z : noise_z + block_noise_size_z,
        ]
        cnt -= 1
    return x


def generate_pair(batch, seg, config: ModelGenesisConfig):
    batch_size = batch.shape[0]

    y = batch
    x = copy.deepcopy(y)

    # Create noised input image
    for n in range(batch_size):
        # Flip -- Need to do it for both if we do it for one
        x[n], y[n], seg[n] = data_augmentation(x[n], y[n], seg[n], config.flip_rate)
        # Local Shuffle Pixel
        x[n] = local_pixel_shuffling(x[n], prob=config.local_rate)
        # Apply non-Linear transformation with an assigned probability
        x[n] = nonlinear_transformation(x[n], config.nonlinear_rate)
        # Inpainting & Outpainting
        if random.random() < config.paint_rate:
            if random.random() < config.inpaint_rate:
                # Inpainting
                x[n] = image_in_painting(x[n])
            else:
                # Outpainting
                x[n] = image_out_painting(x[n])

    return {"input": x, "target": y, "seg": seg}


class ModelGenesisTransform(AbstractTransform):
    def __init__(self, config: ModelGenesisConfig = ModelGenesisConfig()):
        self.config = config

    def __call__(self, **data_dict):
        return generate_pair(data_dict["data"], data_dict["seg"], self.config)
