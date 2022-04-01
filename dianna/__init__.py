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
__version__ = "0.4.1"


def explain_image(model_or_function, input_data, method, labels=(1,), **kwargs):
    """
    Explain an image (input_data) given a model and a chosen method.

    Args:
        model_or_function (callable or str): The function that runs the model to be explained _or_
                                             the path to a ONNX model on disk.
        input_data (np.ndarray): Image data to be explained
        method (string): One of the supported methods: RISE, LIME or KernelSHAP
        labels (tuple): Labels to be explained

    Returns:
        One heatmap (2D array) per class.

    """
    if method == "KernelSHAP":
        # To avoid Access Violation on Windows with SHAP:
        from onnx_tf.backend import prepare  # pylint: disable=import-outside-toplevel,unused-import
    explainer = _get_explainer(method, kwargs)
    explain_image_kwargs = utils.get_kwargs_applicable_to_function(explainer.explain_image, kwargs)
    return explainer.explain_image(model_or_function, input_data, labels, **explain_image_kwargs)


def explain_text(model_or_function, input_data, method, labels=(1,), **kwargs):
    """
    Explain text (input_data) given a model and a chosen method.

    Args:
        model_or_function (callable or str): The function that runs the model to be explained _or_
                                             the path to a ONNX model on disk.
        input_data (string): Text to be explained
        method (string): One of the supported methods: RISE or LIME
        labels (tuple): Labels to be explained

    Returns:
        List of (word, index of word in raw text, importance for target class) tuples.

    """
    explainer = _get_explainer(method, kwargs)
    explain_text_kwargs = utils.get_kwargs_applicable_to_function(explainer.explain_text, kwargs)
    return explainer.explain_text(model_or_function, input_data, labels, **explain_text_kwargs)


def _get_explainer(method, kwargs):
    method_submodule = importlib.import_module(f'dianna.methods.{method.lower()}')
    method_class = getattr(method_submodule, method)
    method_kwargs = utils.get_kwargs_applicable_to_function(method_class.__init__, kwargs)
    return method_class(**method_kwargs)
