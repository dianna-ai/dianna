import streamlit as st
from _model_utils import load_labels
from _model_utils import load_model
from _models_ts import explain_ts_dispatcher
from _models_ts import predict
from _shared import _get_method_params
from _shared import _get_top_indices_and_labels
from _shared import _methods_checkboxes
from _shared import add_sidebar_logo
from _shared import data_directory
from _ts_utils import open_timeseries
from dianna.visualization import plot_timeseries


add_sidebar_logo()

st.title('Time series explanation')

st.error(
    'Time series explanation is still work in progress and not yet functioning!'
)

with st.sidebar:
    st.header('Input data')

    load_example = st.checkbox('Load example data', key='ts_example_check')

    ts_file = st.file_uploader('Select input data',
                               type=(),
                               disabled=load_example)

    ts_model_file = st.file_uploader('Select model',
                                     type='onnx',
                                     disabled=load_example)

    ts_label_file = st.file_uploader('Select labels',
                                     type='txt',
                                     disabled=load_example)

    if load_example:
        ts_file = (data_directory / 'xxx.suffix')
        ts_model_file = (data_directory / 'xxx.onnx')
        ts_label_file = (data_directory / 'xxx.txt')

if not (ts_file and ts_model_file and ts_label_file):
    st.info('Add your input data in the left panel to continue')
    st.stop()

ts_data, _ = open_timeseries(ts_file)

model = load_model(ts_model_file)
serialized_model = model.SerializeToString()

labels = load_labels(ts_label_file)

choices = ()
methods = _methods_checkboxes(choices=choices)

method_params = _get_method_params(methods)

with st.spinner('Predicting class'):
    predictions = predict(model=model, ts_data=ts_data)

top_indices, top_labels = _get_top_indices_and_labels(predictions=predictions,
                                                      labels=labels)

weight = 0.9 / len(methods)
column_spec = [0.1, *[weight for _ in methods]]

_, *columns = st.columns(column_spec)
for col, method in zip(columns, methods):
    with col:
        st.header(method)

for index, label in zip(top_indices, top_labels):
    index_col, *columns = st.columns(column_spec)

    with index_col:
        st.header(label)

    for col, method in zip(columns, methods):
        kwargs = method_params[method].copy()
        kwargs['labels'] = [index]

        func = explain_ts_dispatcher[method]

        with col:
            with st.spinner(f'Running {method}'):
                segments = func(serialized_model, ..., **kwargs)

            fig = plot_timeseries(...)

            st.pyplot(fig)
