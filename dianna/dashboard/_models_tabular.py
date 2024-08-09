import numpy as np
import streamlit as st
from dianna import explain_tabular
from dianna.utils.onnx_runner import SimpleModelRunner


@st.cache_data
def predict(*, model, tabular_input):
    model_runner = SimpleModelRunner(model)
    predictions = model_runner(tabular_input.reshape(1,-1).astype(np.float32))
    return predictions


@st.cache_data
def _run_rise_tabular(_model, table, **kwargs):
    relevances = explain_tabular(
        _model,
        table,
        method='RISE',
        **kwargs,
    )
    return relevances


@st.cache_data
def _run_lime_tabular(_model, table, **kwargs):
    relevances = explain_tabular(
        _model,
        table,
        method='LIME',
        **kwargs,
    )
    return relevances

@st.cache_data
def _run_kernelshap_tabular(model, image, i, **kwargs):
    # Kernelshap interface is different. Write model to temporary file.
    with tempfile.NamedTemporaryFile() as f:
        f.write(model)
        f.flush()
        relevances = explain_tabular(
            _model,
            table,
            method='KernelSHAP',
            **kwargs,
        )
    return relevances


explain_text_dispatcher = {
    'RISE': _run_rise_tabular,
    'LIME': _run_lime_tabular,
    'KernelSHAP': _run_kernelshap_tabular
}
