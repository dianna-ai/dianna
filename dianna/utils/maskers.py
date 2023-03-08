import warnings

import numpy as np


def generate_masks(input_data: np.array, number_of_masks: int, p_keep: float = 0.5):
    """
    Generate a set of masks given a probability of keeping any time step unmasked.

    Args:
        input_data:
        number_of_masks:
        p_keep: the probability that any value remains unmasked.

    Returns:
    Single array containing all masks where the first dimension represents the batch.
    """
    masked_data_shape = [number_of_masks] + list(input_data.shape)
    masks = np.zeros(masked_data_shape, dtype=np.bool)
    for i in range(number_of_masks):
        series_length = input_data.shape[0]
        number_of_steps_masked = _determine_number_of_steps_masked(p_keep, series_length)
        steps_to_mask = np.random.choice(series_length, number_of_steps_masked, False)
        masked_value = 1
        masks[i, steps_to_mask] = masked_value
    return masks


def mask_data(data, masks, mask_type='mean'):
    """
    Mask data given using a set of masks.

    Args:
        data:
        masks: an array with shape [number_of_masks] + data.shape
        mask_type:

    Returns:
    Single array containing all masked input where the first dimension represents the batch.
    """
    number_of_masks = masks.shape[0]
    input_data_batch = np.repeat(np.expand_dims(data, 0), number_of_masks, axis=0)
    result = np.empty(input_data_batch.shape)
    result[~masks] = input_data_batch[~masks]
    result[masks] = _get_mask_value(data, mask_type)
    return result


def _get_mask_value(data: np.array, mask_type: str) -> int:
    """Calculates a masking value of the given type for the data."""
    if mask_type == 'mean':
        return np.mean(data)
    raise ValueError(f'Unknown mask_type selected: {mask_type}')


def _determine_number_of_steps_masked(p_keep: float, series_length: int) -> int:
    user_requested_steps = int(np.round(series_length * (1 - p_keep)))
    if user_requested_steps == series_length:
        warnings.warn('Warning: p_keep chosen too low. Continuing with masking 1 time step per mask.')
        return series_length - 1
    if user_requested_steps == 0:
        warnings.warn('Warning: p_keep chosen too high. Continuing with leaving 1 time step unmasked per mask.')
        return 1
    return user_requested_steps
