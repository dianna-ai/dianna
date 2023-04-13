import numpy as np
import streamlit as st
from _model_utils import data_directory
from _model_utils import load_labels
from _model_utils import load_model
from _models_text import explain_text_dispatcher
from _models_text import predict
from _movie_model import MovieReviewsModelRunner
from _text_utils import format_word_importances


st.title('Text explanation')

with st.sidebar:
    st.header('Input data')

    load_example = st.checkbox('Load example data', key='text_example_check')

    text_input = st.text_input('Input string', disabled=load_example)

    if text_input:
        st.write(text_input)

    text_model_file = st.file_uploader('Select model',
                                       type='onnx',
                                       disabled=load_example)

    text_label_file = st.file_uploader('Select labels',
                                       type='txt',
                                       disabled=load_example)

    if load_example:
        text_input = 'The movie started out great but the ending was dissappointing'
        text_model_file = data_directory / 'movie_review_model.onnx'
        text_label_file = data_directory / 'labels_text.txt'

if not (text_input and text_model_file and text_label_file):
    st.info('Add your input data in the left panel to continue')
    st.stop()

model = load_model(text_model_file)
serialized_model = model.SerializeToString()

labels = load_labels(text_label_file)

methods = []
for col, method in zip(st.columns(2), ('RISE', 'LIME')):
    with col:
        if st.checkbox(method):
            methods.append(method)

if not methods:
    st.info('Select a method to continue')
    st.stop()

kws = {'RISE': {}, 'LIME': {}}

with st.expander('Click to modify method parameters'):
    for method, col in zip(methods, st.columns(len(methods))):
        with col:
            st.header(method)
            if method == 'RISE':
                kws['RISE']['n_masks'] = st.number_input('Number of masks',
                                                         value=1000)
                kws['RISE']['feature_res'] = st.number_input(
                    'Feature resolution', value=6)
                kws['RISE']['p_keep'] = st.number_input(
                    'Probability to be kept unmasked', value=0.1)

            if method == 'LIME':
                kws['LIME']['rand_state'] = st.number_input('Random state',
                                                            value=2)

model_runner = MovieReviewsModelRunner(serialized_model)

with st.spinner('Predicting class'):
    predictions = predict(model=serialized_model, text_input=text_input)

predicted_class = labels[np.argmax(predictions)]
predicted_index = labels.index(predicted_class)

st.info(f'The predicted class is: {predicted_class}')

columns = st.columns(len(methods))

for col, method in zip(columns, methods):
    kwargs = kws[method].copy()
    kwargs['method'] = method
    kwargs['labels'] = [predicted_index]

    func = explain_text_dispatcher[method]

    with col:
        st.header(method)

        with st.spinner(f'Running {method}'):
            relevances = func(model_runner, text_input, **kwargs)

        html = format_word_importances(text_input, relevances[0])
        st.write(html, unsafe_allow_html=True)
