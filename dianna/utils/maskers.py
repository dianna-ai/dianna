import heapq
import warnings
from typing import Iterable
from typing import Union
import numpy as np
from numpy import ndarray
from skimage.transform import resize


def generate_tabular_masks(
    input_data_shape: tuple[int],
    number_of_masks: int,
    p_keep: float = 0.5,
):
    """Generator function to create masks for tabular data.

    Args:
        input_data_shape: Shape of the tabular data to be masked.
        number_of_masks: Number of masks to generate.
        p_keep: probability that any value should remain unmasked.

    Returns:
    Single array containing all masks where the first dimension represents the batch.
    """
    instance_length = np.product(input_data_shape)

    for i in range(number_of_masks):
        n_masked = _determine_number_masked(p_keep, instance_length)
        trues = n_masked * [False]
        falses = (instance_length - n_masked) * [True]
        options = trues + falses
        yield np.random.choice(
            a=options,
            size=input_data_shape,
            replace=False,
        )


def generate_timeseries_masks(
    input_data_shape: tuple[int],
    number_of_masks: int,
    feature_res: int = 8,
    p_keep: float = 0.5,
):
    """Generate masks for time series data given a probability of keeping any time step or channel unmasked.

    Args:
        input_data_shape: Shape of the time series data to be masked.
        number_of_masks: Number of masks to generate.
        p_keep: the probability that any value remains unmasked.
        feature_res: Resolution of features in masks.

    Returns:
    Single array containing all masks where the first dimension represents the batch.
    """
    if input_data_shape[-1] == 1:  # univariate data
        return generate_time_step_masks(input_data_shape,
                                        number_of_masks,
                                        p_keep,
                                        number_of_features=feature_res)

    # We have 3 types of mask generation: channel, time step, combined. We take 1/3 of each.
    number_of_channel_masks = number_of_masks // 3
    number_of_time_step_masks = number_of_channel_masks
    number_of_combined_masks = number_of_masks - number_of_time_step_masks - number_of_channel_masks

    time_step_masks = generate_time_step_masks(input_data_shape,
                                               number_of_time_step_masks,
                                               p_keep, feature_res)
    channel_masks = generate_channel_masks(input_data_shape,
                                           number_of_channel_masks, p_keep)

    # Product of two masks: we need sqrt p_keep to ensure correct resulting p_keep
    sqrt_p_keep = np.sqrt(p_keep)
    combined_masks = generate_time_step_masks(
        input_data_shape, number_of_combined_masks,
        sqrt_p_keep, feature_res) * generate_channel_masks(
            input_data_shape, number_of_combined_masks, sqrt_p_keep)

    return np.concatenate([time_step_masks, channel_masks, combined_masks],
                          axis=0)


def generate_channel_masks(input_data_shape: tuple[int], number_of_masks: int,
                           p_keep: float):
    """Generate masks that mask one or multiple channels independently at a time."""
    number_of_channels = input_data_shape[1]
    number_of_channels_masked = _determine_number_masked(
        p_keep, number_of_channels)
    masked_data_shape = [number_of_masks] + list(input_data_shape)
    masks = np.ones(masked_data_shape, dtype=bool)
    for i in range(number_of_masks):
        channels_to_mask = np.random.choice(number_of_channels,
                                            number_of_channels_masked, False)
        masks[i, :, channels_to_mask] = False
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
    input_data_batch = np.repeat(np.expand_dims(data, 0),
                                 number_of_masks,
                                 axis=0)
    result = np.empty(input_data_batch.shape)
    result[masks] = input_data_batch[masks]
    result[~masks] = _get_mask_value(data, mask_type)
    return result


def _get_mask_value(data: np.array, mask_type: object) -> int:
    """Calculates a masking value of the given type for the data."""
    if callable(mask_type):
        return mask_type(data)
    if mask_type == 'mean':
        return np.mean(data)
    raise ValueError(f'Unknown mask_type selected: {mask_type}')


def _determine_number_masked(p_keep: float,
                             series_length: int,
                             element_name='feature') -> int:
    """Determine the number of time steps that need to be masked."""
    mean = series_length * (1 - p_keep)
    floor = np.floor(mean)
    ceil = np.ceil(mean)
    if floor != ceil:
        user_requested_steps = int(
            np.random.choice([floor, ceil], 1, p=[ceil - mean,
                                                  mean - floor])[0])
    else:
        user_requested_steps = int(floor)

    if user_requested_steps >= series_length:
        warnings.warn(
            f'Warning: p_keep chosen too low. Continuing with leaving 1 {element_name} unmasked per mask.'
        )
        return series_length - 1
    if user_requested_steps <= 0:
        warnings.warn(
            f'Warning: p_keep chosen too high. Continuing with masking 1 {element_name} per mask.'
        )
        return 1
    return user_requested_steps


def generate_time_step_masks(input_data_shape: tuple[int],
                             number_of_masks: int, p_keep: float,
                             number_of_features: int):
    """Generate masks that masks complete time steps at a time while masking time steps in a segmented fashion."""
    time_series_length = input_data_shape[0]
    number_of_channels = input_data_shape[1]

    float_masks = generate_interpolated_float_masks_for_timeseries(
        [time_series_length, 1], number_of_masks, number_of_features)[:, :, 0]
    bool_masks = np.empty(shape=float_masks.shape, dtype=bool)

    # Convert float masks to bool masks using a dynamic threshold
    for i in range(float_masks.shape[0]):
        bool_masks[i] = _mask_bottom_ratio(float_masks[i], p_keep)

    return np.repeat(bool_masks, number_of_channels, axis=2)


def _mask_bottom_ratio(float_mask: np.ndarray, p_keep: float) -> np.ndarray:
    """Return a bool mask given a mask of floats and a ratio.

    Return a mask containing bool values where the top p_keep values of the float mask remain unmasked and the rest is
    masked.

    Args:
        float_mask: a mask containing float values
        p_keep: the ratio of keeping cells unmasked

    Returns:
        a mask containing bool
    """
    flat = float_mask.flatten()
    time_indices = list(range(len(flat)))
    number_of_unmasked_cells = _determine_number_masked(
        p_keep, len(time_indices), element_name='time step')
    top_indices = heapq.nsmallest(number_of_unmasked_cells,
                                  time_indices,
                                  key=lambda time_step: flat[time_step])
    flat_mask = np.ones(flat.shape, dtype=bool)
    flat_mask[top_indices] = False
    return flat_mask.reshape(float_mask.shape)


def generate_interpolated_float_masks_for_image(image_shape: Iterable[int],
                                                p_keep: float,
                                                number_of_masks: int,
                                                number_of_features: int):
    """Generates a set of random masks of float values to mask image data.

    Args:
        image_shape (int): Size of a single sample of input data, for images without the channel axis.
        p_keep: ?
        number_of_masks: Number of masks
        number_of_features: Number of features (or blobs) in both dimensions

    Returns:
        The generated masks (np.ndarray)
    """
    grid = np.random.choice(a=(True, False),
                            size=(number_of_masks, number_of_features,
                                  number_of_features),
                            p=(p_keep, 1 - p_keep)).astype('float32')
    mask_shape = image_shape[:2]
    cell_size = np.ceil(np.array(mask_shape) / number_of_features)
    up_size = (number_of_features + 1) * cell_size
    masks = np.empty((number_of_masks, *mask_shape), dtype=np.float32)
    for i in range(masks.shape[0]):
        y_offset = np.random.randint(0, cell_size[0])
        x_offset = np.random.randint(0, cell_size[1])
        # Linear upsampling and cropping
        upscaled = _upscale(grid[i], up_size)
        masks[i, :, :] = upscaled[y_offset:y_offset + image_shape[0],
                                  x_offset:x_offset + image_shape[1]]
    masks = masks.reshape(-1, *mask_shape, 1)
    return masks


def generate_interpolated_float_masks_for_timeseries(
        time_series_shape: Iterable[int], number_of_masks: int,
        number_of_features: int) -> ndarray:
    """Generates a set of random masks to mask time-series data.

    Args:
        time_series_shape (int): Size of a single sample of input time series.
        number_of_masks: Number of masks
        number_of_features: Number of features in the time dimension

    Returns:
        The generated masks (np.ndarray)
    """
    grid = np.random.random(size=(number_of_masks, number_of_features,
                                  1), ).astype('float32')

    masks_shape = (number_of_masks, *time_series_shape)

    if grid.shape == masks_shape:
        masks = grid
    else:
        masks = _project_grids_to_masks(grid, masks_shape)
    return masks.reshape(-1, *time_series_shape, 1)


def _project_grids_to_masks(grids: ndarray, masks_shape: tuple) -> ndarray:
    """Projects a set of (low resolution) grids onto a target resolution masks.

    Args:
        grids: Set of grids with a pattern for each resulting mask
        masks_shape: Resolution of the resulting masks

    Returns:
        Set of masks with specified shape based on the grids
    """
    number_of_features = grids.shape[1]

    mask_len = masks_shape[1]

    masks = np.empty(masks_shape, dtype=np.float32)
    for i_mask in range(masks.shape[0]):
        offset = np.random.random()
        grid = grids[i_mask, :, 0]
        mask = masks[i_mask, :, 0]

        center_keys = []
        for i_mask_step, center_key in enumerate(
                np.linspace(
                    start=offset,
                    stop=number_of_features - 2 +
                    offset,  # See timeseries masking documentation
                    num=mask_len)):
            center_keys.append(center_key)
            ceil_key = int(np.ceil(center_key))
            floor_key = int(np.floor(center_key))
            if ceil_key == floor_key:
                combined_value_from_grid = grid[ceil_key]
            else:
                floor_val = grid[floor_key]
                ceil_val = grid[ceil_key]
                combined_value_from_grid = (
                    ceil_key - center_key) * floor_val + (center_key -
                                                          floor_key) * ceil_val

            mask[i_mask_step] = combined_value_from_grid
        for i_channel in range(masks.shape[-1]):
            masks[
                i_mask, :,
                i_channel] = mask  # Mask all channels with the same time step mask
    return masks


def _upscale(grid_i, up_size):
    """Up samples and crops the grid to result in an array with size up_size."""
    return resize(grid_i,
                  up_size,
                  order=1,
                  mode='reflect',
                  anti_aliasing=False)
