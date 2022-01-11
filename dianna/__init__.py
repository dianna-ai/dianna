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
import logging
from . import methods


logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "DIANNA Team"
__email__ = "dianna-ai@esciencecenter.nl"
__version__ = "0.1.0"


def explain_image(model_or_function, input_data, method, **kwargs):
    """
    Explain an image (input_data) given a model and a chosen method.

    Args:
        model_or_function (callable or str): The function that runs the model to be explained _or_
                                             the path to a ONNX model on disk.
        input_data (np.ndarray): Image data to be explained
        method (string): One of the supported methods: RISE, LIME or KernelSHAP

    Returns:
        One heatmap (2D array) per class.

    """
    return _get_explainer(method, kwargs).explain_image(model_or_function, input_data)


def explain_text(model_or_function, input_data, method, labels=(1,), **kwargs):
    """
    Explain text (input_data) given a model and a chosen method.

    Args:
        model_or_function (callable or str): The function that runs the model to be explained _or_
                                             the path to a ONNX model on disk.
        input_data (string): Text to be explained
        method (string): One of the supported methods: RISE or LIME

    Returns:
        List of (word, index of word in raw text, importance for target class) tuples.

    """
    return _get_explainer(method, kwargs).explain_text(model_or_function, input_data, labels)


def _get_explainer(method, kwargs):
    method_class = getattr(methods, method)
    return method_class(**kwargs)
