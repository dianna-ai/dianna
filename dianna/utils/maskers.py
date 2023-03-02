import numpy as np


def mask_time_steps(input_data: np.array, number_of_masks: int, p_keep: float = 0.5,
                    masking_strategy: str = 'mean') -> np.array:
    """
    Mask random time steps in time series data.

    Args:
        input_data:
        number_of_masks:
        p_keep:
        masking_strategy: strategy for filling masked parts

    Returns:
        masked data
    """
    masked_data = np.zeros([number_of_masks] + list(input_data.shape)) + input_data
    series_length = input_data.shape[0]
    number_of_steps_masked = _determine_number_of_steps_masked(p_keep, series_length)
    for i in range(number_of_masks):
        steps_to_mask = np.random.choice(series_length, number_of_steps_masked, False)
        masked_value = np.mean(input_data)
        masked_data[i, steps_to_mask] = masked_value
    return masked_data


def _determine_number_of_steps_masked(p_keep: float, series_length: int) -> int:
    user_requested_steps = int(np.round(series_length * (1 - p_keep)))
    if user_requested_steps == series_length:
        return series_length - 1
    if user_requested_steps == 0:
        return 1
    return user_requested_steps
