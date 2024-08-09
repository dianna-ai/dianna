import streamlit as st
from _image_utils import open_image
from _model_utils import load_labels
from _model_utils import load_model
from _models_image import explain_image_dispatcher
from _models_image import predict
from _shared import _get_top_indices_and_labels
from _shared import _methods_checkboxes
from _shared import add_sidebar_logo
from _shared import reset_example
from _shared import reset_method
from dianna.utils.downloader import download
from dianna.visualization import plot_image

add_sidebar_logo()

st.title('Tabular data explanation')

st.sidebar.header('Input data')

input_type = st.sidebar.radio(
        label='Select which input to use',
        options = ('Use an example', 'Use your own data'),
        index = None,
        on_change = reset_example,
        key = 'Tabular_input_type'
    )

# Use the examples
if input_type == 'Use an example':
    """load_example = st.sidebar.radio(
        label='Use example',
        options=(''),
        index = None,
        on_change = reset_method,
        key='Tabular_load_example')"""
    st.info("No examples availble yet")
    st.stop()

# Option to upload your own data
if input_type == 'Use your own data':
    tabular_data_file = st.sidebar.file_uploader('Select tabular data', type='csv')
    tabular_model_file = st.sidebar.file_uploader('Select model',
                                            type='onnx')
    tabular_label_file = st.sidebar.file_uploader('Select labels',
                                            type='txt')

if input_type is None:
    st.info('Select which input type to use in the left panel to continue')
    st.stop()

if not (tabular_data_file and tabular_model_file and tabular_label_file):
    st.info('Add your input data in the left panel to continue')
    st.stop()

st.stop()