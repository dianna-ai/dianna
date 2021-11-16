
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
