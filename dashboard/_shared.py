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


def _get_top_indices(predictions, n_top):
    indices = np.array(np.argpartition(predictions, -n_top)[-n_top:])
    indices = indices[np.argsort(predictions[indices])]
    indices = np.flip(indices)
    return indices


def _get_top_indices_and_labels(*, predictions, labels):
    c1, c2 = st.columns(2)

    with c2:
        n_top = st.number_input('Number of top results to show',
                                value=2,
                                min_value=1,
                                max_value=len(labels))

    top_indices = _get_top_indices(predictions, n_top)
    top_labels = [labels[i] for i in top_indices]

    with c1:
        st.metric('Predicted class', top_labels[0])

    return top_indices, top_labels
