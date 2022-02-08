import inspect


def get_function(model_or_function, preprocess_function=None):
    """Converts input to callable function.

    Any keyword arguments are given to the ModelRunner class if the input is a model path.

    Args:
        model_or_function: Can be either model path or function.
            If input is a function, the function is returned unchanged.
        preprocess_function: function to be run to preprocess the data
    """
    from dianna.utils.onnx_runner import SimpleModelRunner  # pylint: disable=import-outside-toplevel
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
    """Returns a subset of `kwargs` of only arguments and keyword arguments of `function`.

    Note that if `function` has a `**kwargs`
    argument, this function should not be necessary (provided the function
    handles `**kwargs` robustly).
    """
    return {key: value for key, value in kwargs.items()
            if key in inspect.getfullargspec(function).args}


def to_xarray(data, axis_labels, required_labels=None):
    """Converts numpy data and axes labels to an xarray object."""
    if isinstance(axis_labels, dict):
        # key = axis index, value = label
        # not all axes have to be present in the input, but we need to provide
        # a name for each axis
        # first ensure negative indices are converted to positive ones
        indices = list(axis_labels.keys())
        for index in indices:
            if index < 0:
                axis_labels[data.ndim + index] = axis_labels.pop(index)
        labels = [axis_labels[index] if index in axis_labels else f'dim_{index}' for index in range(data.ndim)]
    else:
        labels = list(axis_labels)

    # check if the required labels are present
    if required_labels is not None:
        for label in required_labels:
            assert label in labels, f'Required label missing: {label}'

    # import here because it's slow
    import xarray as xr  # pylint: disable=import-outside-toplevel

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


def onnx_model_node_loader(model_path):
    """Onnx model and node labels loader.

    Load onnx model and return the label of its output node and the data type of input node.

    Args:
        model_path (str): The path to a ONNX model on disk.

    Returns:
        loaded onnx model and the label of output node.
    """
    # these imports are done in the function because they are slow
    import onnx  # pylint: disable=import-outside-toplevel
    from onnx_tf.backend import prepare  # pylint: disable=import-outside-toplevel
    onnx_model = onnx.load(model_path)  # load onnx model
    tf_model_rep = prepare(onnx_model, gen_tensor_dict=True)
    label_input_node = tf_model_rep.inputs[0]
    label_output_node = tf_model_rep.outputs[0]
    dtype_input_node = tf_model_rep.tensor_dict[f'{label_input_node}'].dtype

    return onnx_model, dtype_input_node, label_output_node
