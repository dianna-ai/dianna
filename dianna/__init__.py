"""
DIANNA: Deep Insight And Neural Network Analysis.

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
from . import utils


logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "DIANNA Team"
__email__ = "dianna-ai@esciencecenter.nl"
__version__ = "0.7.0"


def explain_image(model_or_function, input_data, method, labels, **kwargs):
    """
    Explain an image (input_data) given a model and a chosen method.

    Args:
        model_or_function (callable or str): The function that runs the model to be explained _or_
                                             the path to a ONNX model on disk.
        input_data (np.ndarray): Image data to be explained
        method (string): One of the supported methods: RISE, LIME or KernelSHAP
        labels (Iterable(int)): Labels to be explained

    Returns:
        One heatmap (2D array) per class.

    """
    if method.upper() == "KERNELSHAP":
        # To avoid Access Violation on Windows with SHAP:
        from onnx_tf.backend import prepare  # pylint: disable=import-outside-toplevel,unused-import
    explainer = _get_explainer(method, kwargs, modality="Image")
    explain_image_kwargs = utils.get_kwargs_applicable_to_function(explainer.explain, kwargs)
    return explainer.explain(model_or_function, input_data, labels, **explain_image_kwargs)


def explain_text(model_or_function, input_text, tokenizer, method, labels, **kwargs):
    """
    Explain text (input_text) given a model and a chosen method.

    Args:
        model_or_function (callable or str): The function that runs the model to be explained _or_
                                             the path to a ONNX model on disk.
        input_text (string): Text to be explained
        tokenizer : Tokenizer class with tokenize and convert_tokens_to_string methods, and mask_token attribute
        method (string): One of the supported methods: RISE or LIME
        labels (Iterable(int)): Labels to be explained

    Returns:
        List of (word, index of word in raw text, importance for target class) tuples.

    """
    explainer = _get_explainer(method, kwargs, modality="Text")
    explain_text_kwargs = utils.get_kwargs_applicable_to_function(explainer.explain, kwargs)
    return explainer.explain(
        model_or_function=model_or_function,
        input_text=input_text,
        labels=labels,
        tokenizer=tokenizer,
        **explain_text_kwargs)


def _get_explainer(method, kwargs, modality):
    try:
        method_submodule = importlib.import_module(f'dianna.methods.{method.lower()}')
    except ImportError as err:
        raise ValueError(f"Method {method} does not exist") from err
    try:
        method_class = getattr(method_submodule, f"{method.upper()}{modality}")
    except AttributeError as err:
        raise ValueError(f"Data modality {modality} is not available for method {method.upper()}") from err
    method_kwargs = utils.get_kwargs_applicable_to_function(method_class.__init__, kwargs)
    return method_class(**method_kwargs)
