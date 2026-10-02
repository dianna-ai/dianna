import tempfile
import numpy as np
import onnxruntime as rt
import streamlit as st
from _image_utils import imagenet_normalize
from dianna import explain_image


def _preprocess_function(*, normalize, axis_labels, scale=1):
    """Model specific preprocessing of a batch of images, applied after the explainer's masking."""
    channels_axis = next(iter(axis_labels)) + 1  # +1 for the batch axis

    def preprocess(batch):
        batch = (batch / scale).astype(np.float32)
        return imagenet_normalize(batch, channels_axis) if normalize else batch

    return preprocess


@st.cache_data
def predict(*, model, image, normalize, axis_labels):
    session = rt.InferenceSession(model.SerializeToString())
    output_node = session.get_outputs()[0].name
    input_node = session.get_inputs()[0].name
    preprocess = _preprocess_function(normalize=normalize,
                                      axis_labels=axis_labels)
    predictions = session.run([output_node],
                              {input_node: preprocess(image[None, ...])})[0]
    return predictions[0]


@st.cache_data
def _run_rise_image(model, image, i, normalize, **kwargs):
    relevances = explain_image(
        model,
        image,
        method='RISE',
        preprocess_function=_preprocess_function(
            normalize=normalize, axis_labels=kwargs['axis_labels']),
        **kwargs,
    )
    return relevances[0]


@st.cache_data
def _run_lime_image(model, image, i, normalize, **kwargs):
    # LIME requires RGB values in [0, 255]
    relevances = explain_image(
        model,
        image * 255,
        preprocess_function=_preprocess_function(
            normalize=normalize, axis_labels=kwargs['axis_labels'], scale=255),
        method='LIME',
        return_masks=False,
        **kwargs,
    )
    return relevances[0]


@st.cache_data
def _run_kernelshap_image(model, image, i, normalize, **kwargs):
    # Kernelshap interface is different. Write model to temporary file.
    with tempfile.NamedTemporaryFile() as f:
        f.write(model)
        f.flush()
        relevances = explain_image(f.name,
                                   image,
                                   method='KernelSHAP',
                                   preprocess_function=_preprocess_function(
                                       normalize=normalize,
                                       axis_labels=kwargs['axis_labels']),
                                   **kwargs)
    return relevances[0]


explain_image_dispatcher = {
    'RISE': _run_rise_image,
    'LIME': _run_lime_image,
    'KernelSHAP': _run_kernelshap_image,
}
