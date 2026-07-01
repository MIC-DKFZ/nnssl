from nnssl.configuration import ANISO_THRESHOLD
import numpy as np
from nnssl.ssl_data.compute_initial_patch_size import get_patch_size


def configure_rotation_dummyDA_mirroring_and_inital_patch_size(patch_size):
    """
    This function is stupid and certainly one of the weakest spots of this implementation. Not entirely sure how we can fix it.
    """
    dim = len(patch_size)
    if dim == 2:
        raise NotImplementedError("We don't do 2d here anymore. Go 3d or go home!")
    elif dim == 3:
        # todo this is not ideal. We could also have patch_size (64, 16, 128) in which case a full 180deg 2d rot would be bad
        # order of the axes is determined by spacing, not image size
        do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
        if do_dummy_2d_data_aug:
            # why do we rotate 180 deg here all the time? We should also restrict it
            rotation_for_DA = {
                "x": (-180.0 / 360 * 2.0 * np.pi, 180.0 / 360 * 2.0 * np.pi),
                "y": (0, 0),
                "z": (0, 0),
            }
        else:
            rotation_for_DA = {
                "x": (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
                "y": (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
                "z": (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
            }
        mirror_axes = (0, 1, 2)
    else:
        raise ValueError("Only 3D supported, but more than 3D given!")

    # todo this function is stupid. It doesn't even use the correct scale range (we keep things as they were in the
    #  old nnunet for now)
    initial_patch_size = get_patch_size(patch_size[-dim:], *rotation_for_DA.values(), (0.85, 1.25))
    if do_dummy_2d_data_aug:
        initial_patch_size[0] = patch_size[0]

    return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes
