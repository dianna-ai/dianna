import streamlit as st
from _model_utils import fill_segmentation
from _model_utils import preprocess_function
from dianna import explain_image


@st.cache_data
def _run_rise_image(model, image, i, **kwargs):
    relevances = explain_image(
        model,
        image,
        **kwargs,
    )
    return relevances[0]


@st.cache_data
def _run_lime_image(model, image, i, **kwargs):
    relevances = explain_image(
        model,
        image * 256,
        preprocess_function=preprocess_function,
        **kwargs,
    )
    return relevances[0]


@st.cache_data
def _run_kernelshap_image(model, image, i, **kwargs):
    st.warning('Kernelshap requires model as a path to the onnx file.')
    return [[0]]
    shap_values, segments_slic = explain_image(model, image, **kwargs)
    return fill_segmentation(shap_values[i][0], segments_slic)


explain_image_dispatcher = {
    'RISE': _run_rise_image,
    'LIME': _run_lime_image,
    'KernelSHAP': _run_kernelshap_image,
}
