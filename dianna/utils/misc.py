
from dianna.utils.onnx_runner import SimpleModelRunner


def get_function(model_or_function):
    """Converts input to callable function.
       Input can be either model path or function.
       If input is a function, the function is returned unchanged.
    """
    if isinstance(model_or_function, str):
        runner = SimpleModelRunner(model_or_function)
    elif callable(model_or_function):
        runner = model_or_function
    else:
        raise TypeError("model_or_function argument must be string (path to model) or function")
    return runner
