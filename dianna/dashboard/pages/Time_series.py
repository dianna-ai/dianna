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
from _shared import label_directory
from _shared import model_directory
from _ts_utils import _convert_to_segments
from _ts_utils import open_timeseries
from dianna.visualization import plot_timeseries, plot_image
import numpy as np

add_sidebar_logo()

st.title('Time series explanation')

st.sidebar.header('Input data')

#load_example_weather = st.sidebar.checkbox('Load weather example', key='TS_weather_example_check')
#load_example_frb = st.sidebar.checkbox('Load FRB example', key='TS_frb_example_check')

load_example = st.sidebar.radio(
    label = "Load example data",
    options = ("Weather", "FRB"),
    index = None,
    key = "TS_load_example"
)

if load_example == None:
    disable_upload = 0
else:
    disable_upload = 1

ts_file = st.sidebar.file_uploader('Select input data',
                                   type='npy',
                                   disabled=disable_upload)

ts_model_file = st.sidebar.file_uploader('Select model',
                                         type='onnx',
                                         disabled=disable_upload)

ts_label_file = st.sidebar.file_uploader('Select labels',
                                         type='txt',
                                         disabled=disable_upload)

if load_example == "Weather":
    ts_file = (data_directory / 'weather_data.npy')
    ts_model_file = (model_directory /
                    'season_prediction_model_temp_max_binary.onnx')
    ts_label_file = (label_directory / 'weather_data_labels.txt')

    st.markdown(
        """This example demonstrates the use of DIANNA 
        on a pre-trained binary classification model for season prediction.
        The input data is the
        [weather prediction dataset](https://zenodo.org/records/5071376).
        This classification model uses time (days) as function of mean temperature to predict if the whole time series is either summer or winter.
        Using a chosen XAI method the relevance scores are displayed on top of the timeseries. The days contributing positively towards the classification decision are indicated in red and those who contribute negatively in blue.
        """)
elif load_example == "FRB":
    ts_file = (data_directory / 'FRB211024.npy')
    ts_model_file = (model_directory /
                    'apertif_frb_dynamic_spectrum_model.onnx')
    ts_label_file = (label_directory / 'apertif_frb_classes.txt')

    # FRB data must be preprocessed
    def preprocess(data):
        # Preprocessing function for FRB use case to get the data in the rightshape
        return np.transpose(data, (0, 2, 1))[..., None].astype(np.float32)
    
    ts_data = open_timeseries(ts_file)
    ts_data_dianna = ts_data.T[None, ...]
    ts_data_model = ts_data[None, ..., None]

    st.markdown(
        """This example demonstrates the use of DIANNA 
        on a pre-trained binary classification model trained to classify Fast Radio Burst (FRB) timeseries data.
        The goal of the pre-trained convolutional neural network is to determine whether or not the input data contains an
        FRB-like signal, whereby the two classes are noise and FRB.
        """)

if not (ts_file and ts_model_file and ts_label_file):
    st.info('Add your input data in the left panel to continue')
    st.stop()

if load_example != "FRB":
    ts_data_model = open_timeseries(ts_file)
    ts_data_dianna = ts_data_model

model = load_model(ts_model_file)
serialized_model = model.SerializeToString()

labels = load_labels(ts_label_file)

choices = ('LIME', 'RISE')
methods = _methods_checkboxes(choices=choices, key='TS_cb_')

method_params = _get_method_params(methods, key='TS_params_')

with st.spinner('Predicting class'):
    predictions = predict(model=serialized_model, ts_data=ts_data_model)

top_indices, top_labels = _get_top_indices_and_labels(
    predictions=predictions[0], labels=labels)

weight = 0.9 / len(methods)
column_spec = [0.1, *[weight for _ in methods]]

_, *columns = st.columns(column_spec)
for col, method in zip(columns, methods):
    col.header(method)

for index, label in zip(top_indices, top_labels):
    index_col, *columns = st.columns(column_spec)
    index_col.markdown(f'##### {label}')

    for col, method in zip(columns, methods):
        kwargs = method_params[method].copy()
        kwargs['labels'] = [index]
        if load_example == "FRB":
            kwargs['_preprocess_function'] = preprocess

        func = explain_ts_dispatcher[method]

        with col:
            with st.spinner(f'Running {method}'):
                explanation = func(serialized_model, ts_data=ts_data_dianna, **kwargs)

            if load_example == "FRB":
                # normalize FRB data and get rid of last dimension
                fig, _ = plot_image(explanation[0, :, ::-1].T,
                                    ((ts_data + np.min(ts_data)) / (np.max(ts_data) + np.min(ts_data)))[::-1]
                                    )
            else:
                segments = _convert_to_segments(explanation)

                fig, _ = plot_timeseries(range(len(ts_data_dianna[0])), ts_data_dianna[0], segments)

            st.pyplot(fig)

st.stop()