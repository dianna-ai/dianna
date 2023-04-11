import numpy as np
import streamlit as st
from _image_utils import open_image
from _model_utils import load_labels
from _model_utils import load_model
from _models_image import explain_image_dispatcher
from _models_image import get_top_indices
from _models_image import predict
from dianna.visualization import plot_image


st.title("Dianna's dashboard")

with st.sidebar:
    st.header('Input data')

    image_file = st.file_uploader('Select image', type=('png', 'jpg', 'jpeg'))

    if image_file:
        st.image(image_file)

    model_file = st.file_uploader('Select model', type='onnx')

    label_file = st.file_uploader('Select labels', type='txt')

if not (image_file and model_file and label_file):
    st.info('Add your input data in the left panel to continue')
    st.stop()

image, _ = open_image(image_file)
assert isinstance(image, np.ndarray)

model = load_model(model_file)
serialized_model = model.SerializeToString()

labels = load_labels(label_file)

methods = st.multiselect('Select XAI methods',
                         options=('RISE', 'KernelSHAP', 'LIME'))

n_top = st.number_input('Number of top results to show',
                        value=2,
                        min_value=0,
                        max_value=len(labels))

if not methods:
    st.info('Select a method to continue')
    st.stop()

tabs = st.tabs(methods)

kws = {'RISE': {}, 'KernelSHAP': {}, 'LIME': {}}

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

        if method == 'KernelSHAP':
            with c1:
                kws['KernelSHAP']['nsamples'] = st.number_input(
                    'Number of samples', value=1000)
                kws['KernelSHAP']['background'] = st.number_input('Background',
                                                                  value=0)
            with c2:
                kws['KernelSHAP']['n_segments'] = st.number_input(
                    'Number of segments', value=200)
                kws['KernelSHAP']['sigma'] = st.number_input('σ', value=0)

        if method == 'LIME':
            with c1:
                kws['LIME']['rand_state'] = st.number_input('Random state',
                                                            value=2)

with st.spinner('Predicting class'):
    predictions = predict(model=model, image=image)

predicted_class = labels[np.argmax(predictions)]

st.info(f'The predicted class is: {predicted_class}')

top_indices = get_top_indices(predictions, n_top)
top_labels = [labels[i] for i in top_indices]

# check which axis is color channel
original_data = image[:, :, 0] if image.shape[2] <= 3 else image[1, :, :]
axis_labels = {2: 'channels'} if image.shape[2] <= 3 else {0: 'channels'}

columns = st.columns(len(methods))

for col, method in zip(columns, methods):
    kwargs = kws[method].copy()
    kwargs['method'] = method
    kwargs['axis_labels'] = axis_labels

    func = explain_image_dispatcher[method]

    with col:
        st.header(method)

        for index, label in enumerate(top_labels):
            with st.spinner(f'Running {method}'):
                kwargs['labels'] = [top_indices[index]]
                heatmap = func(serialized_model, image, index, **kwargs)

            st.write(f'index={index}, label={label}')

            fig = plot_image(heatmap,
                             original_data=original_data,
                             heatmap_cmap='bwr',
                             show_plot=False)
            st.pyplot(fig)
