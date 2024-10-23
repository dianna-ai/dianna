import base64
import sys
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
    """Upload DIANNA logo to sidebar element."""
    st.sidebar.image(str(data_directory / 'logo.png'))


def _methods_checkboxes(*, choices: Sequence, key):
    """Get methods from a horizontal row of checkboxes and the corresponding parameters."""
    n_choices = len(choices)
    methods = []
    method_params = {}

    # Create a container for the message
    message_container = st.empty()

    for col, method in zip(st.columns(n_choices), choices):
        with col:
            if st.checkbox(method, key=f'{key}_{method}'):
                methods.append(method)
                with st.expander(f'Click to modify {method} parameters'):
                    method_params[method] = _get_params(method, key=f'{key}_param')

    if not methods:
        # Put the message in the container above
        message_container.info('Select a method to continue')
        st.stop()

    return methods, method_params


def _get_params(method: str, key):
    if method == 'RISE':
        n_masks = 1000
        fr = 8
        pkeep = 0.1
        if 'FRB' in key:
            n_masks = 5000
            fr = 16
        elif 'Tabular' in key:
            pkeep = 0.5
        elif 'Weather' in key:
            n_masks = 10000
        elif 'Digits' in key:
            n_masks = 5000
        return {
            'n_masks':
            st.number_input('Number of masks', value=n_masks, key=f'{key}_{method}_nmasks'),
            'feature_res':
            st.number_input('Feature resolution', value=fr, key=f'{key}_{method}_fr'),
            'p_keep':
            st.number_input('Probability to be kept unmasked', value=pkeep, key=f'{key}_{method}_pkeep'),
        }

    elif method == 'KernelSHAP':
        if 'Tabular' in key:
            return {'training_data_kmeans': st.number_input('Training data kmeans', value=5,
                                                            key=f'{key}_{method}_training_data_kmeans'),
            }
        else:
            return {
                'nsamples': st.number_input('Number of samples', value=1000, key=f'{key}_{method}_nsamp'),
                'background': st.number_input('Background', value=0, key=f'{key}_{method}_background'),
                'n_segments': st.number_input('Number of segments', value=200, key=f'{key}_{method}_nseg'),
                'sigma': st.number_input('σ', value=0, key=f'{key}_{method}_sigma'),
            }

    elif method == 'LIME':
        if 'Tabular' in key:
            return {
                'random_state': st.number_input('Random state', value=2, key=f'{key}_{method}_rs'),
                'num_samples': st.number_input('Number of samples', value=2000, key=f'{key}_{method}_ns')
            }
        else:
            return {
                'random_state': st.number_input('Random state', value=2, key=f'{key}_{method}_rs'),
                'num_features': st.number_input('Number of features', 999, key=f'{key}_{method}_rf'),
                'num_samples': st.number_input('Number of samples', value=2000, key=f'{key}_{method}_ns')
            }

    else:
        raise ValueError(f'No such method: {method}')


def _get_top_indices(predictions, n_top):
    indices = np.array(np.argpartition(predictions, -n_top)[-n_top:])
    indices = indices[np.argsort(predictions[indices])]
    indices = np.flip(indices)
    return indices


def _get_top_indices_and_labels(*, predictions, labels):
    cols = st.columns(4)

    if labels is not None:
        with cols[-1]:
            n_top = st.number_input('Number of top classes to show',
                                    value=1,
                                    min_value=1,
                                    max_value=len(labels))

        top_indices = _get_top_indices(predictions, n_top)
        top_labels = [labels[i] for i in top_indices]

        with cols[0]:
            st.metric('Predicted class:', top_labels[0])
    else:
        # If not a classifier, only return the predicted value
        top_indices = top_labels = " "
        with cols[0]:
            st.metric('Predicted value:', f"{predictions[0]:.2f}")

    return top_indices, top_labels

def reset_method():
    # Clear selection
    for k in st.session_state.keys():
        if '_param' in k:
            st.session_state.pop(k)
        elif '_cb' in k:
            st.session_state[k] = False

def reset_example():
    # Clear selection
    for k in st.session_state.keys():
        if '_load_' in k:
            st.session_state.pop(k)
