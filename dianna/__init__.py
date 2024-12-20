"""DIANNA: Deep Insight And Neural Network Analysis.

Modern scientific challenges are often tackled with (Deep) Neural Networks (DNN).
Despite their high predictive accuracy, DNNs lack inherent explainability. Many DNN
users, especially scientists, do not harvest DNNs power because of lack of trust and
understanding of their working.

Meanwhile, the eXplainable AI (XAI) methods offer some post-hoc interpretability and
insight into the DNN reasoning. This is done by quantifying the relevance of individual
features (image pixels, words in text, etc.) with respect to the prediction. These
"relevance heatmaps" indicate how the network has reached its decision directly in the
input modality (images, text, speech etc.) of the data.

There are many Open Source Software (OSS) implementations of these methods, alas,
supporting a single DNN format and the libraries are known mostly by the AI experts.
The DIANNA library supports the best XAI methods in the context of scientific usage
providing their OSS implementation based on the ONNX standard and demonstrations on
benchmark datasets. Representing visually the captured knowledge by the AI system can
become a source of (scientific) insights.

See https://github.com/dianna-ai/dianna
"""
import importlib
import logging
from collections.abc import Callable
from collections.abc import Iterable
from typing import Union
import numpy as np
from . import utils

logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = 'DIANNA Team'
__email__ = 'dianna-ai@esciencecenter.nl'
__version__ = '1.7.0'


def explain_timeseries(model_or_function: Union[Callable, str],
                       input_timeseries: np.ndarray, method: str,
                       labels: Iterable[int], **kwargs) -> np.ndarray:
    """Explain timeseries data given a model and a chosen method.

    Args:
        model_or_function (callable or str): The function that runs the model to be explained _or_
                                             the path to a ONNX model on disk.
        input_timeseries (np.ndarray): Timeseries data to be explained
        method (str): One of the supported methods: RISE, LIME or KernelSHAP
        labels (Iterable(int)): Labels to be explained
        kwargs: key word arguments

    Returns:
        np.ndarray: One heatmap per class.

    """
    explainer = _get_explainer(method, kwargs, modality='Timeseries')
    explain_timeseries_kwargs = utils.get_kwargs_applicable_to_function(
        explainer.explain, kwargs)
    for key in explain_timeseries_kwargs.keys():
        kwargs.pop(key)
    if kwargs:
        raise TypeError(f'Error due to following unused kwargs: {kwargs}')
    return explainer.explain(model_or_function, input_timeseries, labels,
                             **explain_timeseries_kwargs)


def explain_image(model_or_function: Union[Callable,
                                           str], input_image: np.ndarray,
                  method: str, labels: Iterable[int], **kwargs) -> np.ndarray:
    """Explain an image (input_data) given a model and a chosen method.

    Args:
        model_or_function (callable or str): The function that runs the model to be explained _or_
                                             the path to a ONNX model on disk.
        input_image (np.ndarray): Image data to be explained
        method (str): One of the supported methods: RISE, LIME or KernelSHAP
        labels (Iterable(int)): Labels to be explained
        kwargs: These keyword parameters are passed on

    Returns:
        np.ndarray: An array containing the heat maps for each class.

    """
    if method.upper() == 'KERNELSHAP':
        # To avoid Access Violation on Windows with SHAP:
        from onnx_tf.backend import prepare  # noqa: F401
    explainer = _get_explainer(method, kwargs, modality='Image')
    explain_image_kwargs = utils.get_kwargs_applicable_to_function(
        explainer.explain, kwargs)
    for key in explain_image_kwargs.keys():
        kwargs.pop(key)
    if kwargs:
        raise TypeError(f'Error due to following unused kwargs: {kwargs}')
    return explainer.explain(model_or_function, input_image, labels,
                             **explain_image_kwargs)


def explain_text(model_or_function: Union[Callable,
                                          str], input_text: str, tokenizer,
                 method: str, labels: Iterable[int], **kwargs) -> list:
    """Explain text (input_text) given a model and a chosen method.

    Args:
        model_or_function (callable or str): The function that runs the model to be explained _or_
                                             the path to a ONNX model on disk.
        input_text (str): Text to be explained
        tokenizer: Tokenizer class with tokenize and convert_tokens_to_string methods, and mask_token attribute
        method (str): One of the supported methods: RISE or LIME
        labels (Iterable(int)): Labels to be explained
        kwargs: These keyword parameters are passed on

    Returns:
        list: List of tuples (word, index of word in raw text, importance for target class) for each class.

    """
    explainer = _get_explainer(method, kwargs, modality='Text')
    explain_text_kwargs = utils.get_kwargs_applicable_to_function(
        explainer.explain, kwargs)
    for key in explain_text_kwargs.keys():
        kwargs.pop(key)
    if kwargs:
        raise TypeError(f'Error due to following unused kwargs: {kwargs}')
    return explainer.explain(
        model_or_function=model_or_function,
        input_text=input_text,
        labels=labels,
        tokenizer=tokenizer,
        **explain_text_kwargs,
    )


def explain_tabular(model_or_function: Union[Callable, str],
                    input_tabular: np.ndarray,
                    method: str,
                    labels=None,
                    **kwargs) -> np.ndarray:
    """Explain tabular (input_text) given a model and a chosen method.

    Args:
        model_or_function (callable or str): The function that runs the model to be explained _or_
                                             the path to a ONNX model on disk.
        input_tabular (np.ndarray): Tabular data to be explained
        method (str): One of the supported methods: RISE, LIME or KernelSHAP
        labels (Iterable(int), optional): Labels to be explained
        kwargs: These keyword parameters are passed on

    Returns:
        np.ndarray: An array containing the heat maps for each class.
    """
    explainer = _get_explainer(method, kwargs, modality='Tabular')
    explain_tabular_kwargs = utils.get_kwargs_applicable_to_function(
        explainer.explain, kwargs)
    for key in explain_tabular_kwargs.keys():
        kwargs.pop(key)
    if kwargs:
        raise TypeError(f'Error due to following unused kwargs: {kwargs}')
    return explainer.explain(
        model_or_function=model_or_function,
        input_tabular=input_tabular,
        labels=labels,
        **explain_tabular_kwargs,
    )


def _get_explainer(method, kwargs, modality):
    try:
        method_submodule = importlib.import_module(
            f'dianna.methods.{method.lower()}_{modality.lower()}')
    except ImportError as err:
        raise ValueError(
            f'Method {method.lower()}_{modality.lower()} does not exist'
        ) from err
    try:
        method_class = getattr(method_submodule, f'{method.upper()}{modality}')
    except AttributeError as err:
        raise ValueError(
            f'Data modality {modality} is not available for method {method.upper()}'
        ) from err
    method_kwargs = utils.get_kwargs_applicable_to_function(
        method_class.__init__, kwargs)
    # Remove used kwargs from list of kwargs passed to the function.
    for key in method_kwargs.keys():
        kwargs.pop(key)
    return method_class(**method_kwargs)
