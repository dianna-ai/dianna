"""Documentation about dianna"""
import logging
from . import methods


logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "DIANNA Team"
__email__ = "dianna-ai@esciencecenter.nl"
__version__ = "0.1.0"


def explain(model_or_function, input_data, method, **kwargs):
    """
    Exampler explainer wrapper
    """
    method_class = getattr(methods, method)
    explainer = method_class(**kwargs)
    return explainer(model_or_function, input_data)
