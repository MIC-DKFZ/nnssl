import numpy as np

from nnssl.imageio.base_reader_writer import BaseReaderWriter
from nnssl.preprocessing.cropping.cropping import crop_to_nonzero

def analyze_case(
    image_files: list[str],
    reader_writer_class: type[BaseReaderWriter],
):
    rw = reader_writer_class()
    images, properties_images = rw.read_images(image_files)
    # ---------------------------- General Fingerprint --------------------------- #
    spacing = properties_images["spacing"]
    shape_before_crop = images.shape[1:]
    shape_after_crop = shape_before_crop  # Can't tell here
    relative_size_after_cropping = np.prod(shape_after_crop) / np.prod(shape_before_crop)

    # we no longer crop and save the cropped images before this is run. Instead we run the cropping on the fly.
    # Downside is that we need to do this twice (once here and once during preprocessing). Upside is that we don't
    # need to save the cropped data anymore. Given that cropping is not too expensive it makes sense to do it this
    # way. This is only possible because we are now using our new input/output interface.
    data_cropped, seg_cropped, bbox = crop_to_nonzero(images, seg=None)
    shape_after_crop = data_cropped.shape[1:]
    relative_size_after_cropping = np.prod(shape_after_crop) / np.prod(shape_before_crop)

    return (
        shape_after_crop,
        spacing,
        relative_size_after_cropping,
    )
