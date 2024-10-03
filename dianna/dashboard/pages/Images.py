import base64
import sys
import streamlit as st
from _image_utils import open_image
from _model_utils import load_labels
from _model_utils import load_model
from _models_image import explain_image_dispatcher
from _models_image import predict
from _shared import _get_top_indices_and_labels
from _shared import _methods_checkboxes
from _shared import add_sidebar_logo
from _shared import reset_example
from _shared import reset_method
from dianna.utils.downloader import download
from dianna.visualization import plot_image

if sys.version_info < (3, 10):
    from importlib_resources import files
else:
    from importlib.resources import files

data_directory = files('dianna.data')
colormap_path = str(data_directory / 'colormap.png')
with open(colormap_path, "rb") as img_file:
    colormap = base64.b64encode(img_file.read()).decode()

def description_explainer(open='open'):
    """Expandable text section with image."""
    return (st.markdown(
            f"""
            <details {open}>
            <summary><b>Description of the explanation</b></summary>

            The explanation is visualised as a **relevance heatmap** overlayed on top of the time series. <br>
            The heatmap consists of the relevance _attributions_ of all individual pixels/super-pixels of the image
            to a **pretrained model**'s classification. <br>
            The attribution heatmap can be computed for any class. <br><br>

            The _bwr (blue white red)_ attribution colormap
            assigns :blue[**blue**] color to negative relevances, **white** color to near-zero values,
            and :red[**red**] color to positive values. <br><br>

            <center><img src="data:image/png;base64,{colormap}" alt="Colormap" width="600" ></center><br>
            </details>
            """,
            unsafe_allow_html=True
           ),
           st.text("")
           )


add_sidebar_logo()

st.title('Explaining Image data classification')

st.sidebar.header('Input data')

input_type = st.sidebar.radio(
        label='Select which input to use',
        options = ('Use an example', 'Use your own data'),
        index = None,
        on_change = reset_example,
        key = 'Image_input_type'
    )

# Use the examples
if input_type == 'Use an example':
    load_example = st.sidebar.radio(
        label='Load example',
        options=('Hand-written digit recognition',),
        index = None,
        on_change = reset_method,
        key='Image_load_example'
        )

    if load_example == 'Hand-written digit recognition':
        image_file = download('digit0.jpg', 'data')
        image_model_file = download('mnist_model_tf.onnx', 'model')
        image_label_file = download('labels_mnist.txt', 'label')

        imagekey = 'Digits_Image_cb'

        st.markdown(
            """
            This example demonstrates the use of DIANNA on explaining a
            [**binary MNIST model**](https://zenodo.org/records/5907177) pretrained on **only** images of
            the hand-written digits 0 and 1. <br>
            The model classifies an image of a hand-written digit as displaying 0 or 1.
            """,
            unsafe_allow_html=True
        )

    else:
        description_explainer()
        st.info('Select an example in the left panel to coninue')
        st.stop()

# Option to upload your own data
if input_type == 'Use your own data':
    load_example = None

    image_file = st.sidebar.file_uploader('Select image',
                                        type=('png', 'jpg', 'jpeg'))

    if image_file:
        st.sidebar.image(image_file)

    image_model_file = st.sidebar.file_uploader('Select model',
                                                type='onnx')

    image_label_file = st.sidebar.file_uploader('Select labels',
                                                type='txt')

    imagekey = 'Image_cb'

if input_type is None:
    description_explainer()
    st.info('Select which input type to use in the left panel to continue')
    st.stop()

if not (image_file and image_model_file and image_label_file):
    description_explainer()
    st.info('Add your input data in the left panel to continue')
    st.stop()

description_explainer("")

image, _ = open_image(image_file)

model = load_model(image_model_file)
serialized_model = model.SerializeToString()

labels = load_labels(image_label_file)

choices = ('RISE', 'KernelSHAP', 'LIME')

st.text("")
st.text("")

with st.container(border=True):
    prediction_placeholder = st.empty()
    methods, method_params = _methods_checkboxes(choices=choices, key=imagekey)

    with st.spinner('Predicting class'):
        predictions = predict(model=model, image=image)

    with prediction_placeholder:
        top_indices, top_labels = _get_top_indices_and_labels(
            predictions=predictions,labels=labels)

st.text("")
st.text("")

# check which axis is color channel
original_data = image[:, :, 0] if image.shape[2] <= 3 else image[1, :, :]
axis_labels = {2: 'channels'} if image.shape[2] <= 3 else {0: 'channels'}

weight = 0.9 / len(methods)
column_spec = [0.1, *[weight for _ in methods]]

_, *columns = st.columns(column_spec)
for col, method in zip(columns, methods):
    col.markdown(f"<h4 style='text-align: center; '>{method}</h4>", unsafe_allow_html=True)

for index, label in zip(top_indices, top_labels):
    index_col, *columns = st.columns(column_spec)
    index_col.markdown(f'##### Class: {label}')

    for col, method in zip(columns, methods):
        kwargs = method_params[method].copy()
        kwargs['axis_labels'] = axis_labels
        kwargs['labels'] = [index]

        func = explain_image_dispatcher[method]

        with col:
            with st.spinner(f'Running {method}'):
                heatmap = func(serialized_model, image, index, **kwargs)

            fig, _ = plot_image(heatmap,
                                original_data=original_data,
                                heatmap_cmap='bwr',
                                show_plot=False)

            st.pyplot(fig)

st.stop()
