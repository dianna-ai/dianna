import tempfile
import streamlit as st
from _model_utils import fill_segmentation
from _model_utils import preprocess_function
from onnx_tf.backend import prepare
from dianna import explain_image


@st.cache_data
def predict(*, model, image):
    output_node = prepare(model, gen_tensor_dict=True).outputs[0]
    predictions = (prepare(model).run(image[None, ...])[str(output_node)])
    return predictions[0]


@st.cache_data
def _run_rise_image(model, image, i, **kwargs):
    relevances = explain_image(
        model,
        image,
        method='RISE',
        **kwargs,
    )
    return relevances[0]


@st.cache_data
def _run_lime_image(model, image, i, **kwargs):
    relevances = explain_image(
        model,
        image * 256,
        preprocess_function=preprocess_function,
        method='LIME',
        **kwargs,
    )
    return relevances[0]


@st.cache_data
def _run_kernelshap_image(model, image, i, **kwargs):
    # Kernelshap interface is different. Write model to temporary file.
    with tempfile.NamedTemporaryFile() as f:
        f.write(model)
        f.flush()
        shap_values, segments_slic = explain_image(f.name,
                                                   image,
                                                   method='KernelSHAP',
                                                   **kwargs)

    return fill_segmentation(shap_values[i][0], segments_slic)


explain_image_dispatcher = {
    'RISE': _run_rise_image,
    'LIME': _run_lime_image,
    'KernelSHAP': _run_kernelshap_image,
}
