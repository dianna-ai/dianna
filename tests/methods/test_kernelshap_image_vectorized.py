"""Check the vectorized KernelSHAP image helpers against the original loop implementations."""
import numpy as np
import pytest
import skimage.segmentation
from dianna.methods.kernelshap_image import KERNELSHAPImage
from dianna.methods.kernelshap_image import _create_heatmaps


def _create_heatmaps_loop(shap_values_list, image_segments):
    """Original (non-vectorized) implementation of _create_heatmaps."""
    per_class_values = np.asarray(shap_values_list)[0].T
    heat_maps = np.zeros((per_class_values.shape[0], *image_segments.shape))
    for i, shap_values_for_class in enumerate(per_class_values):
        class_heat_map = heat_maps[i]
        for index in image_segments.flat:
            class_heat_map[image_segments == index] = shap_values_for_class[
                index - 1]
        heat_maps[i] = class_heat_map
    return heat_maps


def _mask_image_loop(features,
                     segmentation,
                     image,
                     background=None,
                     channels_axis_index=2,
                     datatype=np.float32):
    """Original (non-vectorized) implementation of KERNELSHAPImage._mask_image."""
    if background is None:
        background = image.mean(axis=(0, 1))
    out = np.zeros(
        (features.shape[0], image.shape[0], image.shape[1], image.shape[2]))
    for i in range(features.shape[0]):
        out[i] = image
        for j in range(features.shape[1]):
            if features[i, j] == 0:
                out[i][segmentation == j, :] = background
    if channels_axis_index != 2:
        out = np.transpose(out, (0, 3, 1, 2))
    return out.astype(datatype)


def _segments(image, n_segments):
    return skimage.segmentation.slic(image,
                                     n_segments=n_segments,
                                     compactness=10.0,
                                     sigma=0)


@pytest.mark.parametrize('shape, n_segments, n_classes',
                         [((28, 28, 1), 50, 2), ((64, 48, 3), 100, 5)])
def test_create_heatmaps_matches_loop(shape, n_segments, n_classes):
    """Vectorized heatmaps equal the loop version."""
    rng = np.random.default_rng(0)
    segments = _segments(rng.random(shape), n_segments)
    shap_values = rng.normal(size=(1, np.unique(segments).size, n_classes))

    np.testing.assert_array_equal(
        _create_heatmaps(shap_values, segments),
        _create_heatmaps_loop(shap_values, segments))


@pytest.mark.parametrize('shape', [(28, 28, 1), (64, 48, 3)])
@pytest.mark.parametrize('background', [None, 0, 0.5])
@pytest.mark.parametrize('channels_axis_index', [0, 2])
def test_mask_image_matches_loop(shape, background, channels_axis_index):
    """Vectorized masking equals the loop version."""
    rng = np.random.default_rng(0)
    image = rng.random(shape)
    segments = _segments(image, 50)
    features = rng.integers(0, 2, size=(20, np.unique(segments).size))

    np.testing.assert_array_equal(
        KERNELSHAPImage()._mask_image(features, segments, image, background,
                                      channels_axis_index),
        _mask_image_loop(features, segments, image, background,
                         channels_axis_index))
