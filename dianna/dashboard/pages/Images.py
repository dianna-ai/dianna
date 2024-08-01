import streamlit as st
from _image_utils import open_image
from _model_utils import load_labels
from _model_utils import load_model
from _models_image import explain_image_dispatcher
from _models_image import predict
from _shared import _get_top_indices_and_labels
from _shared import _methods_checkboxes
from _shared import add_sidebar_logo
from _shared import data_directory
from _shared import label_directory
from _shared import model_directory
from _shared import reset_method
from _shared import reset_example
from dianna.visualization import plot_image

add_sidebar_logo()

st.title('Image explanation')

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
        key='Image_load example'
        )

    if load_example == 'Hand-written digit recognition':
        image_file = (data_directory / 'digit0.jpg')
        image_model_file = (model_directory / 'mnist_model_tf.onnx')
        image_label_file = (label_directory / 'labels_mnist.txt')

        st.markdown(
            """
            This example demonstrates the use of DIANNA on a pretrained binary [MNIST](https://yann.lecun.com/exdb/mnist/) model using a hand-written digit images.
            The model predict for an image of a hand-written 0 or 1, which of the two it most likely is.
            This example visualizes the relevance attributions for each pixel/super-pixel by displaying them on top of the input image.
            """
        )
    else:
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

if input_type == None:
    st.info('Select which input type to use in the left panel to continue')
    st.stop()

if not (image_file and image_model_file and image_label_file):
    st.info('Add your input data in the left panel to continue')
    st.stop()

image, _ = open_image(image_file)

model = load_model(image_model_file)
serialized_model = model.SerializeToString()

labels = load_labels(image_label_file)

choices = ('RISE', 'KernelSHAP', 'LIME')

prediction_placeholder = st.empty()

with st.container(border=True):
    methods, method_params = _methods_checkboxes(choices=choices, key='Image_cb_')

    with st.spinner('Predicting class'):
        predictions = predict(model=model, image=image)

with prediction_placeholder:
    top_indices, top_labels = _get_top_indices_and_labels(
        predictions=predictions,labels=labels)

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
