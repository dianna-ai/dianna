import sys
import streamlit as st
from _model_utils import load_labels
from _model_utils import load_model
from _models_text import explain_text_dispatcher
from _models_text import predict
from _movie_model import MovieReviewsModelRunner
from _shared import _get_top_indices_and_labels
from _shared import _methods_checkboxes
from _shared import add_sidebar_logo
from _shared import reset_example
from _shared import reset_method
from dianna.utils.downloader import download
from dianna.visualization.text import highlight_text

if sys.version_info < (3, 10):
    from importlib_resources import files
else:
    from importlib.resources import files
data_directory = files('dianna.data')

add_sidebar_logo()

st.title('Explaining Textual data classification')

st.markdown(
            """
            The explanation is visualised as a **relevance heatmap** overlayed on top of the input text. <br>
            The heatmap consists of the relevance _attributions_ of all individual words of the text
            to a **pretrained model**'s classification. <br>
            The attribution heatmap can be computed for any class.

            To interpret heatmaps, note that the attributions for the LIME and KernelSHAP explainers are bound between
            -1 and 1 and for the RISE explainer between 0 and 1. <br>
            The _bwr (blue white red)_ attribution colormap
            assigns :blue[**blue**] color to negative relevances, **white** color to near-zero values,
            and :red[**red**] color to positive values.
            """,
            unsafe_allow_html=True
           )

st.image(str(data_directory / 'colormap.png'), width = 660)
st.sidebar.header('Input data')

input_type = st.sidebar.radio(
        label='Select which input to use',
        options = ('Use an example', 'Use your own data'),
        index = None,
        on_change = reset_example,
        key = 'Text_input_type'
    )

# Use the examples
if input_type == 'Use an example':
    load_example = st.sidebar.radio(
        label='Use example',
        options=('Movie sentiment classification',),
        index = None,
        on_change = reset_method,
        key='Text_load_example')

    if load_example == 'Movie sentiment classification':
        text_input = st.sidebar.text_input(
            'Input text',
            value='The movie started out great but the ending was disappointing')
        text_model_file = download('movie_review_model.onnx', 'model')
        text_label_file = download('labels_text.txt', 'label')

        st.markdown(
        """
        ***********************************************************************
        This example demonstrates the use of DIANNA on the [Stanford Sentiment
        Treebank dataset](https://nlp.stanford.edu/sentiment/index.html) which
        contains one-sentence movie reviews. <br> A pre-trained [neural network
        classifier](https://zenodo.org/record/5910598) is used, which classifies a movie review
        as positive or negative. <br>
        :blue-background[The input sentence which the model will classify can be modified in
        the editable Input text field in the left panel.]
        """,
        unsafe_allow_html=True
        )
    else:
        st.info('Select an example in the left panel to coninue')
        st.stop()

# Option to upload your own data
if input_type == 'Use your own data':
    text_input = st.sidebar.text_input('Input string')

    if text_input:
        st.sidebar.write(text_input)

    text_model_file = st.sidebar.file_uploader('Select model',
                                            type='onnx')

    text_label_file = st.sidebar.file_uploader('Select labels',
                                            type='txt')

if input_type is None:
    st.info('Select which input type to use in the left panel to continue')
    st.stop()

if not (text_input and text_model_file and text_label_file):
    st.info('Add your input data in the left panel to continue')
    st.stop()

model = load_model(text_model_file)
serialized_model = model.SerializeToString()

labels = load_labels(text_label_file)

choices = ('RISE', 'LIME')

st.text("")
st.text("")

with st.container(border=True):
    prediction_placeholder = st.empty()
    methods, method_params = _methods_checkboxes(choices=choices, key='Text_cb')

    model_runner = MovieReviewsModelRunner(serialized_model)

    with st.spinner('Predicting class'):
        predictions = predict(model=serialized_model, text_input=text_input)

    with prediction_placeholder:
        top_indices, top_labels = _get_top_indices_and_labels(
            predictions=predictions[0], labels=labels)

st.text("")
st.text("")

weight = 0.85 / len(methods)
column_spec = [0.15, *[weight for _ in methods]]

_, *columns = st.columns(column_spec)
for col, method in zip(columns, methods):
    col.markdown(f"<h4 style='text-align: center; '>{method}</h4>", unsafe_allow_html=True)

for index, label in zip(top_indices, top_labels):
    index_col, *columns = st.columns(column_spec)

    index_col.markdown(f'##### Class: {label}')

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

st.stop()
