import streamlit as st
from _image_utils import open_image
from _model_utils import load_labels
from _model_utils import load_model
from _models_image import explain_image_dispatcher
from _models_image import predict
from _shared import _get_method_params
from _shared import _get_top_indices_and_labels
from _shared import _methods_checkboxes
from _shared import add_sidebar_logo
from _shared import data_directory
from _shared import label_directory
from _shared import model_directory
from dianna.visualization import plot_image

add_sidebar_logo()

st.title('Image explanation')

st.sidebar.header('Input data')

load_example = st.sidebar.checkbox('Load example data',
                                   key='Image_example_check')

image_file = st.sidebar.file_uploader('Select image',
                                      type=('png', 'jpg', 'jpeg'),
                                      disabled=load_example)

if image_file:
    st.sidebar.image(image_file)

image_model_file = st.sidebar.file_uploader('Select model',
                                            type='onnx',
                                            disabled=load_example)

image_label_file = st.sidebar.file_uploader('Select labels',
                                            type='txt',
                                            disabled=load_example)

if load_example:
    image_file = (data_directory / 'digit0.jpg')
    image_model_file = (model_directory / 'mnist_model_tf.onnx')
    image_label_file = (label_directory / 'labels_mnist.txt')

if not (image_file and image_model_file and image_label_file):
    st.info('Add your input data in the left panel to continue')
    st.stop()

image, _ = open_image(image_file)

model = load_model(image_model_file)
serialized_model = model.SerializeToString()

labels = load_labels(image_label_file)

choices = ('RISE', 'KernelSHAP', 'LIME')
methods = _methods_checkboxes(choices=choices, key='Image_cb_')

method_params = _get_method_params(methods, key='Image_params_')

with st.spinner('Predicting class'):
    predictions = predict(model=model, image=image)

top_indices, top_labels = _get_top_indices_and_labels(predictions=predictions,
                                                      labels=labels)

# check which axis is color channel
original_data = image[:, :, 0] if image.shape[2] <= 3 else image[1, :, :]
axis_labels = {2: 'channels'} if image.shape[2] <= 3 else {0: 'channels'}

weight = 0.9 / len(methods)
column_spec = [0.1, *[weight for _ in methods]]

_, *columns = st.columns(column_spec)
for col, method in zip(columns, methods):
    col.header(method)

for index, label in zip(top_indices, top_labels):
    index_col, *columns = st.columns(column_spec)
    index_col.markdown(f'##### {label}')

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
