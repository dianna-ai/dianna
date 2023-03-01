import unittest

import numpy
import numpy as np
import pytest

from dianna.utils.maskers import mask_time_steps


def test_mask_has_correct_shape():
    input_data = _get_input_data()
    number_of_masks = 5

    result = _call_masking_function(input_data, number_of_masks=number_of_masks)

    assert result.shape[0] == number_of_masks, 'Should return the correct number of masks.'
    assert result.shape[1:] == input_data.shape, 'Masked instances should each have the correct shape.'


@pytest.mark.parametrize("p_keep_and_expected_rate", [
    (0.0, 0.1),  # Mask all but one
    (0.1, 0.1),
    (0.3, 0.3),
    (0.5, 0.5),
    (0.667, 0.7),
    (0.99, 0.9),  # Mask only 1
])
def test_mask_contains_correct_number_of_unmasked_parts(p_keep_and_expected_rate):
    p_keep, expected_rate = p_keep_and_expected_rate
    input_data = _get_input_data()

    result = _call_masking_function(input_data, p_keep=p_keep)

    assert np.sum(result == input_data) / np.product(result.shape) == expected_rate


def test_mask_contains_correct_parts_are_mean_masked():
    input_data = _get_input_data()
    mean = numpy.mean(input_data)

    result = _call_masking_function(input_data, mask_type='mean')

    masked_parts = result[(result != input_data)]
    assert np.alltrue(masked_parts == mean), f'All elements in {masked_parts} should have value {mean}'


def _get_input_data() -> np.array:
    return np.zeros((10, 1)) + np.arange(10).reshape(10, 1)


def _call_masking_function(input_data, number_of_masks=5, p_keep=.3, mask_type='mean'):
    return mask_time_steps(input_data, number_of_masks, p_keep=p_keep, mask_type=mask_type)
