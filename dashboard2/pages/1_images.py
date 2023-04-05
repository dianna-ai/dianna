import numpy as np
import streamlit as st
from _image_utils import open_image
from _model_utils import fill_segmentation
from _model_utils import load_labels
from _model_utils import load_model
from _model_utils import preprocess_function
from dianna import explain_image
from dianna.visualization import plot_image


st.title("Dianna's dashboard")

with st.sidebar:
    st.header('Input data')

    image_file = st.file_uploader('Image')

    if image_file:
        st.image(image_file)

    model_file = st.file_uploader('Model')

    label_file = st.file_uploader('Labels')

methods = st.multiselect('Select XAI methods',
                         options=('RISE', 'KernelSHAP', 'LIME'))

show_top = st.number_input('Number of top results to show', value=2)

if not (image_file and model_file and label_file):
    st.info('Add your input data in the left panel to continue')
    st.stop()

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

c1, c2 = st.columns(2)

with c1:
    st.button('Update explanation', type='primary')

with c2:
    st.button('Stop explanation', type='secondary')

image, _ = open_image(image_file)
assert isinstance(image, np.ndarray)

model = load_model(model_file)

labels = load_labels(label_file)
labels = [0, 1]

from onnx_tf.backend import prepare


with st.spinner('Preparing data'):
    output_node = prepare(model, gen_tensor_dict=True).outputs[0]

    predictions = (prepare(model).run(image[None, ...])[str(output_node)])
    class_name = labels
    # get the predicted class
    preds = np.array(predictions[0])
    pred_class = class_name[np.argmax(preds)]
    # get the top most likely results
    if show_top > len(class_name):
        show_top = len(class_name)
    # make sure the top results are ordered most to least likely
    ind = np.array(np.argpartition(preds, -show_top)[-show_top:])
    ind = ind[np.argsort(preds[ind])]
    ind = np.flip(ind)
    top = [class_name[i] for i in ind]
    n_rows = len(top)

    if image.shape[2] <= 3:
        z_rise = image[:, :, 0]
        axis_labels = {2: 'channels'}
        colorscale = 'Bluered'
    else:
        z_rise = image[1, :, :]
        axis_labels = {0: 'channels'}
        colorscale = 'jet'

i = 0

if 'RISE' in methods:
    print(method)
    with st.spinner('Running RISE'):
        relevances = explain_image(
            model.SerializeToString(),
            image,
            method=method,
            labels=[ind[i]],
            axis_labels=axis_labels,
            **kws['RISE'],
        )
    fig = plot_image(relevances[0], original_data=z_rise, heatmap_cmap='bwr')
    st.pyplot(fig)

if 'KernelSHAP' in methods:
    print(method)
    with st.spinner('Running KernelSHAP'):
        shap_values, segments_slic = explain_image(model.SerializeToString(),
                                                   image,
                                                   labels=[ind[i]],
                                                   method=method,
                                                   axis_labels=axis_labels,
                                                   **kws['KernelSHAP'])
    fill_segmentation(shap_values[i][0], segments_slic)
    fig = plot_image(relevances[0], original_data=z_rise, heatmap_cmap='bwr')
    st.pyplot(fig)

if 'LIME' in methods:
    print(method)
    with st.spinner('Running LIME'):
        relevances = explain_image(
            model.SerializeToString(),
            image * 256,
            method='LIME',
            axis_labels=axis_labels,
            labels=[ind[i]],
            preprocess_function=preprocess_function,
            **kws['LIME'],
        )
    fig = plot_image(relevances[0], original_data=z_rise, heatmap_cmap='bwr')
    st.pyplot(fig)
