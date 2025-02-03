import numpy as np

import numpy as np
from nnssl.ssl_data.dataloading.volume_fusion_transform import (
    VolumeFusionTransform,
    _mix_image,
    _overlay_bbox,
    _get_bboxes_within_image_bounds,
)


def get_mixing_images(
    n_batch: int = 2, n_channels: int = 3, xyz_size: tuple[int, int, int] = (8, 8, 8)
) -> tuple[np.ndarray, np.ndarray]:
    foreground_images = np.full(tuple([n_batch, n_channels, *xyz_size]), fill_value=0.25)
    background_images = np.full(tuple([n_batch, n_channels, *xyz_size]), fill_value=0.5)
    return foreground_images, background_images


def get_alphas_values(k: int = 3) -> tuple[float]:
    return tuple(np.linspace(0, 1, k))


def half_half_alpha(image: np.ndarray, axis: int) -> np.ndarray:
    mixed_image = np.zeros_like(image)
    _, _, X, Y, Z = image.shape
    if axis == 0:
        mixed_image[:, :, X // 2 :, :, :] = 1
    elif axis == 1:
        mixed_image[:, :, :, Y // 2 :, :] = 1
    elif axis == 2:
        mixed_image[:, :, :, :, Z // 2 :] = 1
    return mixed_image


def test_mix_image_50_50_mix():
    foreground_images = np.full((2, 3, 8, 8, 8), fill_value=0.25)
    background_images = np.full((2, 3, 8, 8, 8), fill_value=0.5)
    mixing_coefficient = np.full_like(foreground_images, fill_value=0.5)
    mixed_images = _mix_image(foreground_images, background_images, mixing_coefficient=mixing_coefficient)
    assert np.allclose(mixed_images, np.full_like(mixed_images, fill_value=0.375))


def test_mix_image_either_or():
    foreground_images = np.full((2, 3, 8, 8, 8), fill_value=0.25)
    background_images = np.full((2, 3, 8, 8, 8), fill_value=0.5)

    random = np.random.random((2, 3, 8, 8, 8))
    mixing_coefficient = np.where(random < 0.5, 0, 1)

    mixed_images = _mix_image(foreground_images, background_images, mixing_coefficient=mixing_coefficient)
    expected_values = set([0.25, 0.5])
    found_values = set(np.unique(mixed_images))
    assert expected_values == found_values


def test_mix_image_either_or_0_05_10():
    foreground_images = np.full((2, 3, 8, 8, 8), fill_value=0)
    background_images = np.full((2, 3, 8, 8, 8), fill_value=1)

    class_ids = np.random.randint(size=(2, 3, 8, 8, 8), low=0, high=3)
    mixing_coefficients = np.zeros_like(class_ids, dtype=np.float32)
    alphas = [0, 0.5, 1.0]
    for i in range(len(alphas)):
        mixing_coefficients[class_ids == i] = alphas[i]

    mixed_images = _mix_image(foreground_images, background_images, mixing_coefficient=mixing_coefficients)
    expected_values = set([0, 0.5, 1.0])
    found_values = set(np.unique(mixed_images))
    assert expected_values == found_values


def test_get_mixing_images():
    foreground_images, background_images = get_mixing_images(n_batch=2, n_channels=3, xyz_size=(8, 8, 8))
    assert foreground_images.shape == (2, 3, 8, 8, 8)
    assert background_images.shape == (2, 3, 8, 8, 8)
    assert np.allclose(foreground_images, np.full((2, 3, 8, 8, 8), fill_value=0.25))
    assert np.allclose(background_images, np.full((2, 3, 8, 8, 8), fill_value=0.5))


def test_get_alphas_values():
    alphas = get_alphas_values(k=3)
    assert len(alphas) == 3
    assert np.allclose(alphas, [0.0, 0.5, 1.0])


def test_half_half_alpha():
    image = np.zeros((2, 3, 8, 8, 8))
    mixed_image = half_half_alpha(image, axis=0)
    assert np.allclose(mixed_image[:, :, 4:, :, :], np.ones((2, 3, 4, 8, 8)))


def test_bounding_boxes_varying_sizes_and_aspects():
    # Use extreme aspect ratios and sizes
    vf_subpatch_size = ((1, 8), (1, 8), (1, 8))  # Minimal to maximal in each dimension
    xs, ys, zs, x_starts, y_starts, z_starts = _get_bboxes_within_image_bounds(10, (8, 8, 8), vf_subpatch_size)
    # Check that all generated sizes and starts are within the valid range
    assert np.all(xs <= 8) and np.all(x_starts + xs <= 8)
    assert np.all(ys <= 8) and np.all(y_starts + ys <= 8)
    assert np.all(zs <= 8) and np.all(z_starts + zs <= 8)


def test_overlapping_bboxes():
    image = np.zeros((1, 8, 8, 8))
    xs, ys, zs, x_starts, y_starts, z_starts = ([4, 5], [4, 5], [4, 5], [0, 3], [0, 3], [0, 3])
    values = [1, 2]
    image = _overlay_bbox(image, values, xs, ys, zs, x_starts, y_starts, z_starts)
    assert np.all(image[3:8, 3:8, 3:8] == 2)  # Assert that the second bbox overwrote the first


def test_bbox_edge_cases():
    image = np.zeros((8, 8, 8))
    # Create a bbox that exactly matches the image dimensions
    xs, ys, zs, x_starts, y_starts, z_starts = ([8], [8], [8], [0], [0], [0])
    values = [1]
    image = _overlay_bbox(image, values, xs, ys, zs, x_starts, y_starts, z_starts)
    assert np.all(image == 1)


def test_full_transform_integration():
    # This would simulate the full workflow using a mock data dictionary
    transform = VolumeFusionTransform(
        vf_mixing_coefficients=[0, 0.5, 1], vf_subpatch_count=(10, 20), vf_subpatch_size=((1, 8), (1, 8), (1, 8))
    )
    data = np.random.rand(4, 3, 8, 8, 8)  # Create some dummy data
    data_dict = {"data": data}
    result = transform(**data_dict)
    assert "input" in result and "target" in result
    assert result["input"].shape == (2, 3, 8, 8, 8)  # Ensure shapes are maintained
    assert result["target"].shape == (2, 1, 8, 8, 8)  # Masks should match the expected shape
