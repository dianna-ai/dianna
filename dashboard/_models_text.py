import streamlit as st
from _movie_model import MovieReviewsModelRunner
from dianna import explain_text
from dianna.utils.tokenizers import SpacyTokenizer


tokenizer = SpacyTokenizer()


@st.cache_data
def predict(*, model, text_input):
    model_runner = MovieReviewsModelRunner(model)
    predictions = model_runner(text_input)
    return predictions


@st.cache_data
def _run_rise_text(_model, text, **kwargs):
    relevances = explain_text(
        _model,
        text,
        tokenizer,
        method='RISE',
        **kwargs,
    )
    return relevances


@st.cache_data
def _run_lime_text(_model, text, **kwargs):
    relevances = explain_text(_model, text, tokenizer, method='LIME', **kwargs)
    return relevances


explain_text_dispatcher = {
    'RISE': _run_rise_text,
    'LIME': _run_lime_text,
}
