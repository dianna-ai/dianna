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