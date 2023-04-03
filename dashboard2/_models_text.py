import streamlit as st
from dianna import explain_text
from dianna.utils.tokenizers import SpacyTokenizer


tokenizer = SpacyTokenizer()

st.cache_data


def _run_rise_text(model, text, **kwargs):
    relevances = explain_text(
        model,
        text,
        tokenizer,
        **kwargs,
    )
    return relevances


st.cache_data


def _run_lime_text(model, text, **kwargs):
    relevances = explain_text(model, text, tokenizer, **kwargs)
    return relevances


explain_text_dispatcher = {
    'RISE': _run_rise_text,
    'LIME': _run_lime_text,
}
