import streamlit as st
from _model_utils import data_directory
from _model_utils import load_labels
from _model_utils import load_model
from _models_text import explain_text_dispatcher
from _models_text import predict
from _movie_model import MovieReviewsModelRunner
from _shared import _methods_checkboxes
from _shared import get_top_indices
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

methods = _methods_checkboxes(choices=('RISE', 'LIME'))

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

c1, _ = st.columns(2)

with c1:
    n_top = st.number_input('Number of top results to show',
                            value=2,
                            min_value=0,
                            max_value=len(labels))

top_indices = get_top_indices(predictions[0], n_top)
top_labels = [labels[i] for i in top_indices]

st.info(f'The predicted class is: {top_labels[0]}')

weight = 0.8 / len(methods)
column_spec = [0.2, *[weight for _ in methods]]

_, *columns = st.columns(column_spec)
for col, method in zip(columns, methods):
    with col:
        st.header(method)

for index, label in enumerate(top_labels):
    index_col, *columns = st.columns(column_spec)

    with index_col:
        st.header(label)

    for col, method in zip(columns, methods):
        kwargs = kws[method].copy()
        kwargs['method'] = method
        kwargs['labels'] = [index]

        func = explain_text_dispatcher[method]

        with col:
            with st.spinner(f'Running {method}'):
                relevances = func(model_runner, text_input, **kwargs)

            html = format_word_importances(text_input, relevances[0])
            st.write(html, unsafe_allow_html=True)
