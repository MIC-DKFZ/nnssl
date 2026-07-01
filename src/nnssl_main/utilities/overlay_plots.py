#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from itertools import repeat
import multiprocessing
from multiprocessing.pool import Pool
from typing import Tuple, Union

import numpy as np
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *
from nnssl.configuration import default_num_processes
from nnssl.imageio.base_reader_writer import BaseReaderWriter
from nnssl.paths import nnssl_raw, nnssl_preprocessed
from nnssl.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnssl.data.utils import get_train_dataset

color_cycle = (
    "000000",
    "4363d8",
    "f58231",
    "3cb44b",
    "e6194B",
    "911eb4",
    "ffe119",
    "bfef45",
    "42d4f4",
    "f032e6",
    "000075",
    "9A6324",
    "808000",
    "800000",
    "469990",
)


def hex_to_rgb(hex: str):
    assert len(hex) == 6
    return tuple(int(hex[i : i + 2], 16) for i in (0, 2, 4))


def generate_image(
    input_image: np.ndarray,
):
    """
    image can be 2d greyscale or 2d RGB (color channel in last dimension!)

    Segmentation must be label map of same shape as image (w/o color channels)

    mapping can be label_id -> idx_in_cycle or None

    returned image is scaled to [0, 255] (uint8)!!!
    """
    # create a copy of image
    image = np.copy(input_image)

    if image.ndim == 2:
        image = np.tile(image[:, :, None], (1, 1, 3))
    elif image.ndim == 3:
        if image.shape[2] == 1:
            image = np.tile(image, (1, 1, 3))
        else:
            raise RuntimeError(
                f"if 3d image is given the last dimension must be the color channels (3 channels). "
                f"Only 2D images are supported. Your image shape: {image.shape}"
            )
    else:
        raise RuntimeError(
            "unexpected image shape. only 2D images and 2D images with color channels (color in "
            "last dimension) are supported"
        )

    # rescale image to [0, 255]
    image = image - image.min()
    image = image / image.max() * 255

    return image.astype(np.uint8)


def plot_overlay(
    image_file: str,
    image_reader_writer: BaseReaderWriter,
    output_file: str,
):
    import matplotlib.pyplot as plt

    image, props = image_reader_writer.read_images((image_file,))
    image = image[0]

    assert image.ndim == 3, "only 3D images/segs are supported"

    selected_slice = image.shape[0] // 2
    # print(image.shape, selected_slice)
    overlay = generate_image(image[selected_slice])
    plt.imsave(output_file, overlay)


def plot_overlay_preprocessed(case_file: str, output_file: str, overlay_intensity: float = 0.6, channel_idx=0):
    import matplotlib.pyplot as plt

    data = np.load(case_file)["data"]

    assert channel_idx < (data.shape[0]), "This dataset only supports channel index up to %d" % (data.shape[0] - 1)

    image = data[channel_idx]
    imshape = image.shape

    selected_slice = imshape[0] // 2
    overlay = generate_image(image[selected_slice])
    plt.imsave(output_file, overlay)


def multiprocessing_plot_overlay(
    list_of_image_files,
    list_of_seg_files,
    image_reader_writer,
    list_of_output_files,
    num_processes=8,
):
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        r = p.starmap_async(
            plot_overlay,
            zip(
                list_of_image_files,
                list_of_seg_files,
                [image_reader_writer] * len(list_of_output_files),
                list_of_output_files,
            ),
        )
        r.get()


def multiprocessing_plot_overlay_preprocessed(
    list_of_case_files, list_of_output_files, num_processes=8, channel_idx=0
):
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        r = p.starmap_async(
            plot_overlay_preprocessed,
            zip(
                list_of_case_files,
                list_of_output_files,
                repeat(channel_idx),
            ),
        )
        r.get()


def generate_overlays_from_raw(
    dataset_name_or_id: Union[int, str],
    output_folder: str,
    num_processes: int = 8,
    channel_idx: int = 0,
):
    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    folder = join(nnssl_raw, dataset_name)
    dataset_json = load_json(join(folder, "dataset.json"))
    dataset = get_train_dataset(folder, dataset_json)

    image_files = [v["images"][channel_idx] for v in dataset.values()]
    seg_files = [v["label"] for v in dataset.values()]

    assert all([isfile(i) for i in image_files])
    assert all([isfile(i) for i in seg_files])

    maybe_mkdir_p(output_folder)
    output_files = [join(output_folder, i + ".png") for i in dataset.keys()]

    image_reader_writer = determine_reader_writer_from_dataset_json(dataset_json, image_files[0])()
    multiprocessing_plot_overlay(image_files, seg_files, image_reader_writer, output_files, num_processes)


def generate_overlays_from_preprocessed(
    dataset_name_or_id: Union[int, str],
    output_folder: str,
    num_processes: int = 8,
    channel_idx: int = 0,
    configuration: str = None,
    plans_identifier: str = "nnUNetPlans",
):
    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    folder = join(nnssl_preprocessed, dataset_name)
    if not isdir(folder):
        raise RuntimeError("run preprocessing for that task first")

    plans = load_json(join(folder, plans_identifier + ".json"))
    if configuration is None:
        if "3d_fullres" in plans["configurations"].keys():
            configuration = "3d_fullres"
        else:
            configuration = "2d"
    data_identifier = plans["configurations"][configuration]["data_identifier"]
    preprocessed_folder = join(folder, data_identifier)

    if not isdir(preprocessed_folder):
        raise RuntimeError(
            f"Preprocessed data folder for configuration {configuration} of plans identifier "
            f"{plans_identifier} ({dataset_name}) does not exist. Run preprocessing for this "
            f"configuration first!"
        )

    identifiers = [i[:-4] for i in subfiles(preprocessed_folder, suffix=".npz", join=False)]

    output_files = [join(output_folder, i + ".png") for i in identifiers]
    image_files = [join(preprocessed_folder, i + ".npz") for i in identifiers]

    maybe_mkdir_p(output_folder)
    multiprocessing_plot_overlay_preprocessed(
        image_files,
        output_files,
        num_processes=num_processes,
        channel_idx=channel_idx,
    )


def entry_point_generate_overlay():
    import argparse

    parser = argparse.ArgumentParser(
        "Plots png overlays of the slice with the most foreground. Note that this " "disregards spacing information!"
    )
    parser.add_argument("-d", type=str, help="Dataset name or id", required=True)
    parser.add_argument("-o", type=str, help="output folder", required=True)
    parser.add_argument(
        "-np",
        type=int,
        default=default_num_processes,
        required=False,
        help=f"number of processes used. Default: {default_num_processes}",
    )
    parser.add_argument(
        "-channel_idx", type=int, default=0, required=False, help="channel index used (0 = _0000). Default: 0"
    )
    parser.add_argument(
        "--use_raw",
        action="store_true",
        required=False,
        help="if set then we use raw data. else " "we use preprocessed",
    )
    parser.add_argument(
        "-p",
        type=str,
        required=False,
        default="nnUNetPlans",
        help="plans identifier. Only used if --use_raw is not set! Default: nnUNetPlans",
    )
    parser.add_argument(
        "-c",
        type=str,
        required=False,
        default=None,
        help="configuration name. Only used if --use_raw is not set! Default: None = "
        "3d_fullres if available, else 2d",
    )

    args = parser.parse_args()

    if args.use_raw:
        generate_overlays_from_raw(
            args.d,
            args.o,
            args.np,
            args.channel_idx,
        )
    else:
        generate_overlays_from_preprocessed(args.d, args.o, args.np, args.channel_idx, args.c, args.p)


if __name__ == "__main__":
    entry_point_generate_overlay()
