import numpy as np
import pytest
from pandas import DataFrame
from dianna.utils.maskers import generate_channel_masks
from dianna.utils.maskers import generate_interpolated_float_masks_for_image
from dianna.utils.maskers import generate_interpolated_float_masks_for_timeseries
from dianna.utils.maskers import generate_masks
from dianna.utils.maskers import generate_time_step_masks
from dianna.utils.maskers import mask_data


def test_mask_has_correct_shape_univariate():
    """Test masked data has the correct shape for a univariate input."""
    input_data = _get_univariate_time_series()
    number_of_masks = 5

    result = generate_masks(input_data, number_of_masks)

    assert result.shape == tuple([number_of_masks] + list(input_data.shape))


def test_mask_has_correct_type_univariate():
    """Test masked data has the correct dtype for a univariate input."""
    input_data = _get_univariate_time_series()
    number_of_masks = 5

    result = generate_masks(input_data, number_of_masks=number_of_masks)

    assert result.dtype == bool


def test_generate_time_step_masks_dtype_multivariate():
    """Test masked data has the correct dtype for a multivariate input."""
    input_data = _get_multivariate_time_series()
    number_of_masks = 5

    result = generate_time_step_masks(input_data,
                                      number_of_masks=number_of_masks,
                                      number_of_features=8,
                                      p_keep=0.5)

    assert result.dtype == bool


def test_generate_segmented_time_step_masks_dtype_multivariate():
    """Test masked data has the correct dtype for a multivariate input."""
    input_data = _get_multivariate_time_series()
    number_of_masks = 5

    result = generate_time_step_masks(input_data,
                                      number_of_masks=number_of_masks,
                                      number_of_features=8,
                                      p_keep=0.5)

    assert result.dtype == bool


def test_mask_has_correct_shape_multivariate():
    """Test masked data has the correct shape for a multivariate input."""
    input_data = _get_multivariate_time_series()
    number_of_masks = 5

    result = _call_masking_function(input_data,
                                    number_of_masks=number_of_masks)

    assert result.shape == tuple([number_of_masks] + list(input_data.shape))


@pytest.mark.parametrize(
    'p_keep_and_expected_rate',
    [
        (0.1, 0.1),  # Mask all but one
        (0.1, 0.1),
        (0.3, 0.3),
        (0.5, 0.5),
        (0.99, 0.9),  # Mask only 1
    ])
def test_mask_contains_correct_number_of_unmasked_parts(
        p_keep_and_expected_rate):
    """Number of unmasked parts should be conforming the given p_keep."""
    p_keep, expected_rate = p_keep_and_expected_rate
    input_data = _get_univariate_time_series()

    result = _call_masking_function(input_data, p_keep=p_keep)

    assert np.sum(result == input_data) / np.product(
        result.shape) == expected_rate


def test_mask_contains_correct_parts_are_mean_masked():
    """All parts that are masked should now contain the mean of the input."""
    input_data = _get_univariate_time_series()
    mean = np.mean(input_data)

    result = _call_masking_function(input_data, mask_type='mean')

    masked_parts = result[(result != input_data)]
    assert np.alltrue(
        masked_parts ==
        mean), f'All elements in {masked_parts} should have value {mean}'


def _get_univariate_time_series(num_steps=10) -> np.array:
    """Get some univariate test data."""
    return np.zeros(
        (num_steps, 1)) + np.arange(num_steps).reshape(num_steps, 1)


def _get_multivariate_time_series(number_of_channels: int = 6) -> np.array:
    """Get some multivariate test data."""
    return np.row_stack([
        np.zeros((10, number_of_channels)),
        np.ones((10, number_of_channels))
    ])


def _call_masking_function(
    input_data,
    number_of_masks=5,
    p_keep=.3,
    mask_type='mean',
    feature_res=5,
):
    """Helper function with some defaults to call the code under test."""
    masks = generate_masks(input_data,
                           number_of_masks,
                           feature_res,
                           p_keep=p_keep)
    return mask_data(input_data, masks, mask_type=mask_type)


def test_channel_mask_has_correct_shape_multivariate():
    """Tests the output has the correct shape."""
    number_of_masks = 15
    input_data = _get_multivariate_time_series()

    result = generate_channel_masks(input_data, number_of_masks, 0.5)

    assert result.shape == tuple([number_of_masks] + list(input_data.shape))


def test_channel_mask_has_does_not_contain_conflicting_values():
    """Tests that only complete channels are masked."""
    number_of_masks = 15
    input_data = _get_multivariate_time_series()

    result = generate_channel_masks(input_data, number_of_masks, 0.5)

    unexpected_results = []
    for mask_i, mask in enumerate(result):
        for channel_i in range(mask.shape[-1]):
            channel = mask[:, channel_i]
            value = channel[0]
            if (not value) in channel:
                unexpected_results.append(
                    f'Mask {mask_i} contains conflicting values in channel {channel_i}. Channel: {channel}'
                )
    assert not unexpected_results


def test_channel_mask_masks_correct_number_of_cells():
    """Tests whether the correct fraction of cells is masked."""
    number_of_masks = 1
    input_data = _get_multivariate_time_series(number_of_channels=10)
    p_keep = 0.3

    result = generate_channel_masks(input_data, number_of_masks, p_keep)

    assert result.sum() / np.product(result.shape) == p_keep


def test_masking_has_correct_shape_multivariate():
    """Test for the correct output shape for the general masking function."""
    number_of_masks = 15
    input_data = _get_multivariate_time_series()

    result = generate_masks(input_data, number_of_masks)

    assert result.shape == tuple([number_of_masks] + list(input_data.shape))


def test_masking_univariate_leaves_anything_unmasked():
    """Tests that something remains unmasked and some parts are masked for the univariate case."""
    number_of_masks = 1
    input_data = _get_univariate_time_series()

    result = generate_masks(input_data, number_of_masks)

    assert np.any(result)
    assert np.any(~result)


def test_masking_keep_first_instance():
    """First instance must be the original data for Lime timeseries.

    Required by `lime_base` explainer, the first instance of masked (or perturbed)
    data must be the original instance.

    More details can be found in the code:
    https://github.com/marcotcr/lime/blob/fd7eb2e6f760619c29fca0187c07b82157601b32/lime/lime_base.py#L148
    """
    input_data = _get_multivariate_time_series()
    number_of_masks = 5
    masks = generate_masks(input_data, number_of_masks, p_keep=0.9)
    masks[0, :, :] = 1.0
    masked = mask_data(input_data, masks, mask_type="mean")
    assert np.array_equal(masked[0, :, :], input_data)


@pytest.mark.parametrize('num_steps', range(3, 20))
def test_masks_approximately_correct_number_of_masked_parts_per_time_step(
        num_steps):
    """Number of unmasked parts should be conforming the given p_keep."""
    p_keep = 0.5
    number_of_masks = 500
    input_data = _get_univariate_time_series(num_steps=num_steps)

    masks = generate_masks(input_data,
                           number_of_masks=number_of_masks,
                           feature_res=num_steps,
                           p_keep=p_keep)[:, :, 0]

    masks_mean = DataFrame(masks).mean()
    print('\n')
    print(masks_mean)
    assert np.allclose(masks_mean, p_keep, atol=0.1)


@pytest.mark.parametrize('num_steps', range(5, 20))
def test_masks_approximately_correct_number_of_masked_parts_per_time_step_projected(
        num_steps):
    """Number of unmasked parts should be conforming the given p_keep."""
    p_keep = 0.5
    number_of_masks = 500
    input_data = _get_univariate_time_series(num_steps=num_steps)

    masks = generate_masks(input_data,
                           number_of_masks=number_of_masks,
                           feature_res=6,
                           p_keep=p_keep)[:, :, 0]
    print_univariate_masks(masks[:5])

    masks_mean = DataFrame(masks).mean()
    print('\n')
    print(masks_mean)
    assert np.allclose(masks_mean, p_keep, atol=0.1)


def print_univariate_masks(masks: np.ndarray):
    """Print univariate masks to console for inspection."""
    print('\n')
    for mask in masks:
        steps = ['1' if s else '0' for s in mask]
        print(' '.join(steps))


@pytest.mark.parametrize('num_steps_and_num_features', [
    (10, 10),
    (3, 3),
    (30, 30),
    (30, 10),
    (50, 5),
])
def test_generate_interpolated_mean_float_masks_for_timeseries(
        num_steps_and_num_features):
    """Mean of float masks should be 0.5."""
    num_steps, num_features = num_steps_and_num_features

    expected_mean = 0.5
    number_of_masks = 500
    input_data = _get_univariate_time_series(num_steps=num_steps)

    masks = generate_interpolated_float_masks_for_timeseries(
        input_data.shape,
        number_of_masks=number_of_masks,
        number_of_features=num_features,
    )[:, :, 0, 0]

    masks_mean = np.mean(masks, axis=0)
    print('\n')
    print(masks_mean)
    assert np.allclose(masks_mean, expected_mean, atol=0.1)


@pytest.mark.parametrize('num_steps_and_num_features', [
    (10, 10),
    (3, 3),
    (30, 30),
    (30, 10),
    (50, 5),
])
def test_generate_interpolated_mean_float_masks_for_image(
        num_steps_and_num_features):
    """Mean of float masks should be conforming the given p_keep."""
    num_steps, num_features = num_steps_and_num_features

    p_keep = 0.3
    number_of_masks = 500
    image_shape = (224, 224, 3)

    masks = generate_interpolated_float_masks_for_image(
        image_shape,
        number_of_masks=number_of_masks,
        number_of_features=num_features,
        p_keep=p_keep,
    )[:, :, 0, 0]

    masks_mean = np.mean(masks, axis=0)
    print('\n')
    print(masks_mean)
    assert np.allclose(masks_mean, p_keep, atol=0.1)
