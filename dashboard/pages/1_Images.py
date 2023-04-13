import numpy as np
import streamlit as st
from _image_utils import open_image
from _model_utils import data_directory
from _model_utils import load_labels
from _model_utils import load_model
from _models_image import explain_image_dispatcher
from _models_image import predict
from _shared import _methods_checkboxes
from _shared import get_top_indices
from dianna.visualization import plot_image


st.title('Image explanation')

with st.sidebar:
    st.header('Input data')

    load_example = st.checkbox('Load example data', key='image_example_check')

    image_file = st.file_uploader('Select image',
                                  type=('png', 'jpg', 'jpeg'),
                                  disabled=load_example)

    if image_file:
        st.image(image_file)

    image_model_file = st.file_uploader('Select model',
                                        type='onnx',
                                        disabled=load_example)

    image_label_file = st.file_uploader('Select labels',
                                        type='txt',
                                        disabled=load_example)

    if load_example:
        image_file = (data_directory / 'digit0.png')
        image_model_file = (data_directory / 'mnist_model_tf.onnx')
        image_label_file = (data_directory / 'labels_mnist.txt')

if not (image_file and image_model_file and image_label_file):
    st.info('Add your input data in the left panel to continue')
    st.stop()

image, _ = open_image(image_file)
assert isinstance(image, np.ndarray)

model = load_model(image_model_file)
serialized_model = model.SerializeToString()

labels = load_labels(image_label_file)

methods = _methods_checkboxes(choices=('RISE', 'KernelSHAP', 'LIME'))

kws = {'RISE': {}, 'KernelSHAP': {}, 'LIME': {}}

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

            if method == 'KernelSHAP':
                kws['KernelSHAP']['nsamples'] = st.number_input(
                    'Number of samples', value=1000)
                kws['KernelSHAP']['background'] = st.number_input('Background',
                                                                  value=0)
                kws['KernelSHAP']['n_segments'] = st.number_input(
                    'Number of segments', value=200)
                kws['KernelSHAP']['sigma'] = st.number_input('σ', value=0)

            if method == 'LIME':
                kws['LIME']['rand_state'] = st.number_input('Random state',
                                                            value=2)

c1, _ = st.columns(2)

with c1:
    n_top = st.number_input('Number of top results to show',
                            value=2,
                            min_value=0,
                            max_value=len(labels))

with st.spinner('Predicting class'):
    predictions = predict(model=model, image=image)

top_indices = get_top_indices(predictions, n_top)
top_labels = [labels[i] for i in top_indices]

st.info(f'The predicted class is: {top_labels[0]}')

# check which axis is color channel
original_data = image[:, :, 0] if image.shape[2] <= 3 else image[1, :, :]
axis_labels = {2: 'channels'} if image.shape[2] <= 3 else {0: 'channels'}

weight = 0.9 / len(methods)
column_spec = [0.1, *[weight for _ in methods]]

_, *columns = st.columns(column_spec)
for col, method in zip(columns, methods):
    with col:
        st.header(method)

for index, label in zip(top_indices, top_labels):
    index_col, *columns = st.columns(column_spec)

    with index_col:
        st.header(label)

    for col, method in zip(columns, methods):
        kwargs = kws[method].copy()
        kwargs['method'] = method
        kwargs['axis_labels'] = axis_labels

        func = explain_image_dispatcher[method]

        with col:
            with st.spinner(f'Running {method}'):
                kwargs['labels'] = [index]
                heatmap = func(serialized_model, image, index, **kwargs)

            fig = plot_image(heatmap,
                             original_data=original_data,
                             heatmap_cmap='bwr',
                             show_plot=False)

            st.pyplot(fig)
