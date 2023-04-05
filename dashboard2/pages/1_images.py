import streamlit as st
from _image_utils import open_image
from _shared import make_rise_plot


st.title("Dianna's dashboard")

with st.sidebar:
    st.header('Input data')

    image_file = st.file_uploader('Image')

    if image_file:
        st.image(image_file)

    model = st.file_uploader('Model')

    labels = st.file_uploader('Labels')

methods = st.multiselect('Select XAI methods',
                         options=('Rise', 'KernelSHAP', 'LIME'))

st.number_input('Number of top results to show', value=2)

if not methods:
    st.info('Select a method to continue')
    st.stop()

tabs = st.tabs(methods)

for method, tab in zip(methods, tabs):
    with tab:
        c1, c2 = st.columns(2)
        if method == 'Rise':

            with c1:
                rise_n_masks = st.number_input('Number of masks', value=1000)
                rise_feat_res = st.number_input('Feature resolution', value=6)
            with c2:
                rise_unmask_prob = st.number_input(
                    'Probability to be kept unmasked', value=0.1)

        if method == 'KernelSHAP':

            with c1:
                kshap_n_samples = st.number_input('Number of samples',
                                                  value=1000)
                kshap_background = st.number_input('Background', value=0)
            with c2:
                kshap_n_segments = st.number_input('Number of segments',
                                                   value=200)
                ksnap_sigma = st.number_input('σ', value=0)

        if method == 'LIME':

            with c1:
                lime_rand_state = st.number_input('Random state', value=2)

c1, c2 = st.columns(2)

with c1:
    st.button('Update explanation', type='primary')

with c2:
    st.button('Stop explanation', type='secondary')

image, _ = open_image(image_file)

import numpy as np


assert isinstance(image, np.ndarray)

rise_fig = make_rise_plot(
    image=image,
    model=model,
    labels=labels,
    rise_n_masks=rise_n_masks,
    rise_feat_res=rise_feat_res,
    rise_unmask_prob=rise_unmask_prob,
)

st.pyplot(rise_fig)
