import base64
import sys
from typing import Any
from typing import Dict
from typing import Sequence
import numpy as np
import streamlit as st

if sys.version_info < (3, 10):
    from importlib_resources import files
else:
    from importlib.resources import files

data_directory = files('dianna.data')
model_directory = files('dianna.models')
label_directory = files('dianna.labels')

@st.cache_data
def get_base64_of_bin_file(png_file):
    with open(png_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def build_markup_for_logo(
    png_file,
    background_position='10% 10%',
    margin_top='0%',
    image_width='60%',
    image_height='',
    padding_top='70px'
):
    binary_string = get_base64_of_bin_file(png_file)
    return f"""
            <style>
                [data-testid="stSidebarNav"] {{
                    background-image: url("data:image/png;base64,{binary_string}");
                    background-repeat: no-repeat;
                    background-position: {background_position};
                    margin-top: {margin_top};
                    background-size: {image_width} {image_height};
                    padding-top: {padding_top};
                }}
            </style>
            """


def add_sidebar_logo():
    """Based on: https://stackoverflow.com/a/73278825."""
    png_file = data_directory / 'logo.png'
    logo_markup = build_markup_for_logo(png_file)
    st.markdown(
        logo_markup,
        unsafe_allow_html=True,
    )


def _methods_checkboxes(*, choices: Sequence, key):
    """Get methods from a horizontal row of checkboxes
    and the corresponding parameters."""
    n_choices = len(choices)
    methods = []
    method_params = {}

    # Create a container for the message
    message_container = st.empty()

    for col, method in zip(st.columns(n_choices), choices):
        with col:
            if st.checkbox(method, key=key + method):
                methods.append(method)
                with st.expander(f'Click to modify {method} parameters'):
                    method_params[method] = _get_params(method, key=key)

    if not methods:
        # Put the message in the container above
        message_container.info('Select a method to continue')
        st.stop()
    return methods, method_params


def _get_params(method: str, key):
    if method == 'RISE':
        return {
            'n_masks':
            st.number_input('Number of masks', value=1000, key=key + method + 'nmasks'),
            'feature_res':
            st.number_input('Feature resolution', value=6, key=key + method + 'fr'),
            'p_keep':
            st.number_input('Probability to be kept unmasked', value=0.1, key=key + method + 'pkeep'),
        }

    elif method == 'KernelSHAP':
        return {
            'nsamples': st.number_input('Number of samples', value=1000, key=key + method + 'nsamp'),
            'background': st.number_input('Background', value=0, key=key + method + 'background'),
            'n_segments': st.number_input('Number of segments', value=200, key=key + method + 'nseg'),
            'sigma': st.number_input('σ', value=0, key=key + method + 'sigma'),
        }

    elif method == 'LIME':
        return {
            'random_state': st.number_input('Random state', value=2, key=key + method + 'rs'),
        }

    else:
        raise ValueError(f'No such method: {method}')


def _get_top_indices(predictions, n_top):
    indices = np.array(np.argpartition(predictions, -n_top)[-n_top:])
    indices = indices[np.argsort(predictions[indices])]
    indices = np.flip(indices)
    return indices


def _get_top_indices_and_labels(*, predictions, labels):
    c1, c2 = st.columns(2)

    with c2:
        n_top = st.number_input('Number of top classes to show',
                                value=2,
                                min_value=1,
                                max_value=len(labels))

    top_indices = _get_top_indices(predictions, n_top)
    top_labels = [labels[i] for i in top_indices]

    with c1:
        st.metric('Predicted class', top_labels[0])

    return top_indices, top_labels

def reset_method():
    # Clear selection
    for k in st.session_state.keys():
        if '_cb_' in k:
            st.session_state[k] = False
        if 'params' in k:
            st.session_state.pop(k)

def reset_example():
    # Clear selection
    for k in st.session_state.keys():
        if '_load_' in k:
            st.session_state.pop(k)