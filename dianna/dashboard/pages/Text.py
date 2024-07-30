import streamlit as st
from _model_utils import load_labels
from _model_utils import load_model
from _models_text import explain_text_dispatcher
from _models_text import predict
from _movie_model import MovieReviewsModelRunner
from _shared import _get_method_params
from _shared import _get_top_indices_and_labels
from _shared import _methods_checkboxes
from _shared import add_sidebar_logo
from _shared import label_directory
from _shared import model_directory
from dianna.visualization.text import highlight_text

add_sidebar_logo()

st.title('Text explanation')

st.sidebar.header('Input data')

load_example = st.sidebar.checkbox('Load example data',
                                key='Text_example_check')

text_input = st.sidebar.text_input('Input string', disabled=load_example)

if text_input:
    st.sidebar.write(text_input)

text_model_file = st.sidebar.file_uploader('Select model',
                                           type='onnx',
                                           disabled=load_example)

text_label_file = st.sidebar.file_uploader('Select labels',
                                           type='txt',
                                           disabled=load_example)

if load_example:
    text_input = 'The movie started out great but the ending was dissappointing'
    text_model_file = model_directory / 'movie_review_model.onnx'
    text_label_file = label_directory / 'labels_text.txt'

if not (text_input and text_model_file and text_label_file):
    st.info('Add your input data in the left panel to continue')
    st.stop()

model = load_model(text_model_file)
serialized_model = model.SerializeToString()

labels = load_labels(text_label_file)

choices = ('RISE', 'LIME')
methods = _methods_checkboxes(choices=choices, key='Text_cb_')

method_params = _get_method_params(methods, key='Text_params_')

model_runner = MovieReviewsModelRunner(serialized_model)

with st.spinner('Predicting class'):
    predictions = predict(model=serialized_model, text_input=text_input)

top_indices, top_labels = _get_top_indices_and_labels(
    predictions=predictions[0], labels=labels)

weight = 0.85 / len(methods)
column_spec = [0.15, *[weight for _ in methods]]

_, *columns = st.columns(column_spec)
for col, method in zip(columns, methods):
    col.header(method)

for index, label in zip(top_indices, top_labels):
    index_col, *columns = st.columns(column_spec)

    index_col.markdown(f'##### {label}')

    for col, method in zip(columns, methods):
        kwargs = method_params[method].copy()
        kwargs['labels'] = [index]

        func = explain_text_dispatcher[method]

        with col:
            with st.spinner(f'Running {method}'):
                relevances = func(model_runner, text_input, **kwargs)

            fig, _ = highlight_text(explanation=relevances[0], show_plot=False)
            st.pyplot(fig)

    # add some white space to separate rows
    st.markdown('')
