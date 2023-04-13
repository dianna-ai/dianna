from typing import Sequence
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
