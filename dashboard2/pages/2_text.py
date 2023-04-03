from pathlib import Path
import numpy as np
import streamlit as st
from _model_utils import load_labels
from _model_utils import load_model
from _models_text import explain_text_dispatcher
from _movie_model import MovieReviewsModelRunner
from _text_utils import format_word_importances
import dianna


st.title("Dianna's dashboard")

with st.sidebar:
    st.header('Input data')

    text_input = st.text_input('Input string')

    if text_input:
        st.write(text_input)

    model_file = st.file_uploader('Model', type='onnx')

    label_file = st.file_uploader('Labels', type='txt')

methods = st.multiselect('Select XAI methods', options=('RISE', 'LIME'))

if not (text_input and model_file and label_file):
    st.info('Add your input data in the left panel to continue')
    st.stop()

if not methods:
    st.info('Select a method to continue')
    st.stop()

tabs = st.tabs(methods)

kws = {'RISE': {}, 'LIME': {}}

for method, tab in zip(methods, tabs):
    with tab:
        c1, c2 = st.columns(2)
        if method == 'RISE':
            with c1:
                kws['RISE']['n_masks'] = st.number_input('Number of masks',
                                                         value=1000)
                kws['RISE']['feature_res'] = st.number_input(
                    'Feature resolution', value=6)
            with c2:
                kws['RISE']['p_keep'] = st.number_input(
                    'Probability to be kept unmasked', value=0.1)

        if method == 'LIME':
            with c1:
                kws['LIME']['rand_state'] = st.number_input('Random state',
                                                            value=2)

model = load_model(model_file)
serialized_model = model.SerializeToString()

labels = load_labels(label_file)

with st.spinner('Preparing data'):
    # TODO: Re-organize this mess
    dianna_root_dir = Path(dianna.__file__).parents[1]
    word_vector_path = dianna_root_dir / 'tutorials' / 'data' / 'movie_reviews_word_vectors.txt'

    model_runner = MovieReviewsModelRunner(serialized_model,
                                           word_vector_path,
                                           max_filter_size=5)
    predictions = model_runner(text_input)
    pred_class = labels[np.argmax(predictions)]
    pred_idx = tuple(labels).index(pred_class)

    st.info(f'The predicted class is: {pred_class}')

columns = st.columns(len(methods))

for col, method in zip(columns, methods):
    kwargs = kws[method].copy()
    kwargs['method'] = method
    kwargs['labels'] = [pred_idx]

    func = explain_text_dispatcher[method]

    with col:
        st.header(method)

        with st.spinner(f'Running {method}'):
            relevances = func(model_runner, text_input, **kwargs)

        weight_map = {r[0]: r[2] for r in relevances[0]}

        html = format_word_importances(text_input, weight_map)
        st.write(html, unsafe_allow_html=True)
