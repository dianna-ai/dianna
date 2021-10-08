
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
