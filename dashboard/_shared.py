from typing import Sequence
import numpy as np
import streamlit as st


def _methods_checkboxes(*, choices: Sequence):
    """Get methods from a horizontal row of checkboxes."""
    n_choices = len(choices)
    methods = []
    for col, method in zip(st.columns(n_choices), choices):
        with col:
            if st.checkbox(method):
                methods.append(method)

    if not methods:
        st.info('Select a method to continue')
        st.stop()

    return methods


def get_top_indices(predictions, n_top):
    indices = np.array(np.argpartition(predictions, -n_top)[-n_top:])
    indices = indices[np.argsort(predictions[indices])]
    indices = np.flip(indices)
    return indices
