import base64
import sys
import numpy as np
import streamlit as st
from _model_utils import load_labels
from _model_utils import load_model
from _models_ts import explain_ts_dispatcher
from _models_ts import predict
from _shared import _get_top_indices_and_labels
from _shared import _methods_checkboxes
from _shared import add_sidebar_logo
from _shared import reset_example
from _shared import reset_method
from _ts_utils import _convert_to_segments
from _ts_utils import open_timeseries
from matplotlib import pyplot as plt
from dianna.utils.downloader import download
from dianna.visualization import plot_timeseries

if sys.version_info < (3, 10):
    from importlib_resources import files
else:
    from importlib.resources import files

data_directory = files('dianna.data')
colormap_path = str(data_directory / 'colormap.png')
with open(colormap_path, "rb") as img_file:
    colormap = base64.b64encode(img_file.read()).decode()

def description_explainer(open='open'):
    """Expandable text section with image."""
    return (st.markdown(
            f"""
            <details {open}>
            <summary><b>Description of the explanation</b></summary>

            The explanation is visualised as a **relevance heatmap** overlayed on top of the time series. <br>
            The heatmap consists of the relevance _attributions_ of all individual data points per time moment
            of the series to a **pretrained model**'s classification. <br>
            The attribution heatmap can be computed for any class.

            The _bwr (blue white red)_ attribution colormap
            assigns :blue[**blue**] color to negative relevances, **white** color to near-zero values,
            and :red[**red**] color to positive values.

            <img src="data:image/png;base64,{colormap}" alt="Colormap" width="600" ><br>
            </details>
            """,
            unsafe_allow_html=True
           ),
           st.text("")
           )

st.title('Explaining Time series data classification')

add_sidebar_logo()

st.sidebar.header('Input data')

input_type = st.sidebar.radio(
        label='Select which input to use',
        options = ('Use an example', 'Use your own data'),
        index = None,
        on_change = reset_example,
        key = 'TS_input_type'
    )

# Use the examples
if input_type == 'Use an example':
    load_example = st.sidebar.radio(
        label='Select example',
        options = ('Season prediction from temperature: warm or cold?',
                   'Scientific case - radio astronomy: Fast Radio Burst (FRB) detection'),
        index = None,
        on_change = reset_method,
        key = 'TS_load_example'
    )

    if load_example == "Season prediction from temperature: warm or cold?":
        ts_data_file = download('weather_data.npy', 'data')
        ts_model_file = download(
                        'season_prediction_model_temp_max_binary.onnx', 'model')
        ts_label_file = download('weather_data_labels.txt', 'label')

        param_key = 'Weather_TS_cb'
        description_explainer("")
        st.markdown(
        """
        This example demonstrates the use of DIANNA
        on a pre-trained binary [classification model](https://zenodo.org/records/7543883)
        for season prediction. <br> The input data is the
        [weather prediction dataset](https://zenodo.org/records/5071376).
        The binary classification is simplified to warm or cold (conditionally labelled _summer_ or _winter_) <br>
        The model uses _mean temperature_ as function of time (in days) to predict if the whole
        time series is either from a warm (_summer_) or a cold (_winter_) season.
        """,
        unsafe_allow_html=True
        )
    elif load_example == "Scientific case - radio astronomy: Fast Radio Burst (FRB) detection":
        ts_model_file = download('apertif_frb_dynamic_spectrum_model.onnx', 'model')
        ts_label_file = download('apertif_frb_classes.txt', 'label')
        ts_data_file = download('FRB211024.npy', 'data')

        # FRB data must be preprocessed
        def preprocess(data):
            """Preprocessing function for FRB use case to get the data in the right shape."""
            return np.transpose(data, (0, 2, 1))[..., None].astype(np.float32)

        # Transform FRB data for the model prediction and dianna explanation, which have different
        # requirements for this specific data
        ts_data = open_timeseries(ts_data_file)
        ts_data_explainer = ts_data.T[None, ...]
        ts_data_predictor = ts_data[None, ..., None]

        param_key = 'FRB_TS_cb'
        description_explainer("")
        st.markdown(
            """
            This example demonstrates the use of DIANNA
            on a pre-trained [binary model](https://zenodo.org/records/10656614) for classification of
            radio astronomical dynamic spectra, also known as frequency-time data. <br>
            The scientifically relevant goal is to
            determine whether the input data contains a
            Fast Radio Burst (FRB)- like signal. <br>
            The output of the clasisfier is a label for each data point - either noise or FRB.
            """,
            unsafe_allow_html=True
            )
    else:
        description_explainer()
        st.info('Select an example in the left panel to coninue')
        st.stop()


# Option to upload your own data
if input_type == 'Use your own data':
    load_example = None

    ts_data_file = st.sidebar.file_uploader('Select input data',
                                    type='npy')

    ts_model_file = st.sidebar.file_uploader('Select model',
                                            type='onnx')

    ts_label_file = st.sidebar.file_uploader('Select labels',
                                            type='txt')

    param_key = 'TS_cb'

    if not (ts_data_file and ts_model_file and ts_label_file):
        description_explainer()
        st.info('Add your input data in the left panel to continue')
        st.stop()
    else:
        description_explainer("")

if input_type is None:
    description_explainer()
    st.info('Select which input type to use in the left panel to continue')
    st.stop()

if load_example != "Scientific case - radio astronomy: Fast Radio Burst (FRB) detection":
    # For normal cases, the input data does not need transformation for either the
    # model explainer nor the model predictor
    ts_data_explainer = ts_data_predictor = open_timeseries(ts_data_file)

model = load_model(ts_model_file)
serialized_model = model.SerializeToString()

labels = load_labels(ts_label_file)

if load_example == "Scientific case - radio astronomy: Fast Radio Burst (FRB) detection":
    choices = ('RISE',)
else:
    choices = ('RISE', 'LIME')

st.text("")
st.text("")

with st.container(border=True):
    prediction_placeholder = st.empty()
    methods, method_params = _methods_checkboxes(choices=choices, key=param_key)

    with st.spinner('Predicting class'):
        predictions = predict(model=serialized_model, ts_data=ts_data_predictor)

    with prediction_placeholder:
        top_indices, top_labels = _get_top_indices_and_labels(
            predictions=predictions[0], labels=labels)

st.text("")
st.text("")

weight = 0.9 / len(methods)
column_spec = [0.1, *[weight for _ in methods]]

_, *columns = st.columns(column_spec)
for col, method in zip(columns, methods):
    col.markdown(f"<h4 style='text-align: center; '>{method}</h4>", unsafe_allow_html=True)

for index, label in zip(top_indices, top_labels):
    index_col, *columns = st.columns(column_spec)
    index_col.markdown(f'##### Class: {label}')

    for col, method in zip(columns, methods):
        kwargs = method_params[method].copy()
        kwargs['labels'] = [index]
        if load_example == "Scientific case - radio astronomy: Fast Radio Burst (FRB) detection":
            kwargs['_preprocess_function'] = preprocess

        func = explain_ts_dispatcher[method]

        with col:
            with st.spinner(f'Running {method}'):
                explanation = func(serialized_model, ts_data=ts_data_explainer, **kwargs)

            if load_example == "Scientific case - radio astronomy: Fast Radio Burst (FRB) detection":
                fig, axes = plt.subplots(ncols=2, figsize=(14, 5))
                # FRB: plot original data
                ax = axes[0]
                ax.imshow(ts_data, aspect='auto', origin='lower')
                ax.set_xlabel('Time step')
                ax.set_ylabel('Channel index')
                ax.set_title('Input data')
                # FRB data explanation has to be transposed
                ax = axes[1]
                plot = ax.imshow(explanation[0].T, aspect='auto', origin='lower', cmap='bwr')
                ax.set_xlabel('Time step')
                ax.set_ylabel('Channel index')
                ax.set_title('Explanation')
                fig.colorbar(plot)

            else:
                segments = _convert_to_segments(explanation)

                fig, _ = plot_timeseries(range(len(ts_data_explainer[0])), ts_data_explainer[0], segments)

            st.pyplot(fig)

st.stop()
