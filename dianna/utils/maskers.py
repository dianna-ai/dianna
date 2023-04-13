import warnings
from typing import Union

import numpy as np


def generate_masks(input_data: np.array, number_of_masks: int, p_keep: float = 0.5):
    """Generate masks for time series data given a probability of keeping any time step or channel unmasked.

    Args:
        input_data: Timeseries data to be explained.
        number_of_masks: Number of masks to generate.
        p_keep: the probability that any value remains unmasked.

    Returns:
    Single array containing all masks where the first dimension represents the batch.
    """
    if input_data.shape[-1] == 1:  # univariate data
        return generate_time_step_masks(input_data, number_of_masks, p_keep)

    number_of_channel_masks = number_of_masks // 3
    number_of_time_step_masks = number_of_channel_masks
    number_of_combined_masks = number_of_masks - number_of_time_step_masks - number_of_channel_masks

    time_step_masks = generate_time_step_masks(input_data, number_of_time_step_masks, p_keep)
    channel_masks = generate_channel_masks(input_data, number_of_channel_masks, p_keep)
    number_of_combined_masks = generate_time_step_masks(input_data, number_of_combined_masks,
                                                        p_keep) * generate_channel_masks(input_data,
                                                                                         number_of_combined_masks,
                                                                                         p_keep)

    return np.concatenate([time_step_masks, channel_masks, number_of_combined_masks], axis=0)


def generate_channel_masks(input_data: np.ndarray, number_of_masks: int, p_keep: float):
    """Generate masks that mask one or multiple channels at a time."""
    number_of_channels = input_data.shape[1]
    number_of_channels_masked = _determine_number_masked(p_keep, number_of_channels)
    masked_data_shape = [number_of_masks] + list(input_data.shape)
    masks = np.ones(masked_data_shape, dtype=np.bool)
    for i in range(number_of_masks):
        channels_to_mask = np.random.choice(number_of_channels, number_of_channels_masked, False)
        masks[i, :, channels_to_mask] = False
    return masks


def generate_time_step_masks(input_data: np.ndarray, number_of_masks: int, p_keep: float):
    """Generate masks that mask one or multiple time steps at a time."""
    series_length = input_data.shape[0]
    number_of_steps_masked = _determine_number_masked(p_keep, series_length)
    masked_data_shape = [number_of_masks] + list(input_data.shape)
    masks = np.ones(masked_data_shape, dtype=np.bool)
    for i in range(number_of_masks):
        steps_to_mask = np.random.choice(series_length, number_of_steps_masked, False)
        masks[i, steps_to_mask] = False
    return masks


def mask_data(data: np.array, masks: np.array, mask_type: Union[object, str]):
    """Mask data given using a set of masks.

    Args:
        data: Input data.
        masks: an array with shape [number_of_masks] + data.shape
        mask_type: Masking strategy.

    Returns:
    Single array containing all masked input where the first dimension represents the batch.
    """
    number_of_masks = masks.shape[0]
    input_data_batch = np.repeat(np.expand_dims(data, 0), number_of_masks, axis=0)
    result = np.empty(input_data_batch.shape)
    result[masks] = input_data_batch[masks]
    result[~masks] = _get_mask_value(data, mask_type)
    return result


def _get_mask_value(data: np.array, mask_type: str) -> int:
    """Calculates a masking value of the given type for the data."""
    if callable(mask_type):
        return mask_type(data)
    if mask_type == 'mean':
        return np.mean(data)
    raise ValueError(f'Unknown mask_type selected: {mask_type}')


def _determine_number_masked(p_keep: float, series_length: int) -> int:
    user_requested_steps = int(np.round(series_length * (1 - p_keep)))
    if user_requested_steps == series_length:
        warnings.warn('Warning: p_keep chosen too low. Continuing with leaving 1 time step unmasked per mask.')
        return series_length - 1
    if user_requested_steps == 0:
        warnings.warn('Warning: p_keep chosen too high. Continuing with masking 1 time step per mask.')
        return 1
    return user_requested_steps
