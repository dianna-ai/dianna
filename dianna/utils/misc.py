
import xarray as xr
from dianna.utils.onnx_runner import SimpleModelRunner


def get_function(model_or_function, preprocess_function=None):
    """Converts input to callable function.
       Input can be either model path or function.
       If input is a function, the function is returned unchanged.

       Any keyword arguments are given to the ModelRunner class if the input is a model path.
    """
    if isinstance(model_or_function, str):
        runner = SimpleModelRunner(model_or_function, preprocess_function=preprocess_function)
    elif callable(model_or_function):
        if preprocess_function is None:
            runner = model_or_function
        else:
            def runner(input_data):
                return model_or_function(preprocess_function(input_data))
    else:
        raise TypeError("model_or_function argument must be string (path to model) or function")
    return runner


def get_kwargs_applicable_to_function(function, kwargs):
    """
    Returns a dict that is the subset of `kwargs` for which the keys are
    keyword arguments of `function`. Note that if `function` has a `**kwargs`
    argument, this function should not be necessary (provided the function
    handles `**kwargs` robustly).
    """
    return {key: value for key, value in kwargs.items()
            if key in function.__code__.co_varnames}


def to_xarray(data, axes_labels, required_labels=None):
    """Converts numpy data and axes labels to an xarray object
    """
    if isinstance(axes_labels, dict):
        # key = axis index, value = label
        # not all axes have to be present in the input, but we need to provide
        # a name for each axis
        # first ensure negative indices are converted to positive ones
        indices = list(axes_labels.keys())
        for index in indices:
            if index < 0:
                axes_labels[data.ndim + index] = axes_labels.pop(index)
        labels = [axes_labels[index] if index in axes_labels else f'dim_{index}' for index in range(data.ndim)]
    else:
        labels = list(axes_labels)

    # check if the required labels are present
    if required_labels is not None:
        for label in required_labels:
            assert label in labels, f'Required label missing: {label}'

    return xr.DataArray(data, dims=labels)


def move_axis(data, label, new_position):
    """Moves a named axis to a new position in an xarray DataArray object.

    Args:
        data (DataArray): Object with named axes
        label (str): Name of the axis to move
        new_position (int): Numerical new position of the axis.
                            Negative indices are accepted.

    Returns:
        data with axis in new position
    """
    # find current position of axis
    try:
        pos = data.dims.index(label)
    except ValueError as e:
        raise ValueError(f"Axis name {label} does not exist in input data") from e

    # create list of labels with new ordering
    axis_labels = list(data.dims)
    # the new position will be _before_ the given index, so will fail with a negative index
    # convert to a positive index in that case
    if new_position < 0:
        new_position += len(axis_labels)
    axis_labels.insert(new_position, axis_labels.pop(pos))
    # do the move
    return data.transpose(*axis_labels)
