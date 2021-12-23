"""Documentation about dianna"""
import logging
from . import methods


logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "DIANNA Team"
__email__ = "dianna-ai@esciencecenter.nl"
__version__ = "0.2.1"


def explain_image(model_or_function, input_data, method, **kwargs):
    """
    Exampler explainer wrapper
    """
    return get_explainer(method, kwargs).explain_image(model_or_function, input_data)


def explain_text(model_or_function, input_data, method, labels=(1,), **kwargs):
    """
    Exampler explainer wrapper
    """
    return get_explainer(method, kwargs).explain_text(model_or_function, input_data, labels)


def get_explainer(method, kwargs):
    method_class = getattr(methods, method)
    return method_class(**kwargs)
