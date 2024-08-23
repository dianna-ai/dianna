import tempfile
import onnxruntime as ort
import numpy as np
import streamlit as st
from dianna import explain_tabular


@st.cache_data
def predict(*, model, tabular_input):
    # Make sure that tabular input is provided as float32
    sess = ort.InferenceSession(model)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    onnx_input = {input_name: tabular_input.astype(np.float32)}
    pred_onnx = sess.run([output_name], onnx_input)[0]

    return pred_onnx


@st.cache_data
def _run_rise_tabular(_model, table, training_data, **kwargs):
    # convert streamlit kwarg requirement back to dianna kwarg requirement
    if "_preprocess_function" in kwargs:
        kwargs["preprocess_function"] = kwargs["_preprocess_function"]
        del kwargs["_preprocess_function"]

    def run_model(tabular_input):
        return predict(model=_model, tabular_input=tabular_input)
    
    relevances = explain_tabular(
        run_model,
        table,
        method='RISE',
        training_data=training_data,
        **kwargs,
    )
    return relevances


@st.cache_data
def _run_lime_tabular(_model, table, training_data, _feature_names, **kwargs):
    # convert streamlit kwarg requirement back to dianna kwarg requirement
    if "_preprocess_function" in kwargs:
        kwargs["preprocess_function"] = kwargs["_preprocess_function"]
        del kwargs["_preprocess_function"]

    def run_model(tabular_input):
        return predict(model=_model, tabular_input=tabular_input)
    
    relevances = explain_tabular(
        run_model,
        table,
        method='LIME',
        training_data=training_data,
        feature_names=_feature_names,
        **kwargs,
    )
    return relevances

@st.cache_data
def _run_kernelshap_tabular(model, table, training_data, **kwargs):
    # Kernelshap interface is different. Write model to temporary file.
    with tempfile.NamedTemporaryFile() as f:
        f.write(model)
        f.flush()
        relevances = explain_tabular(f.name,
                table,
                method='KernelSHAP',
                training_data=training_data,
                **kwargs)
    return relevances[0]


explain_tabular_dispatcher = {
    'RISE': _run_rise_tabular,
    'LIME': _run_lime_tabular,
    'KernelSHAP': _run_kernelshap_tabular
}
