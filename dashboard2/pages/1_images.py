import streamlit as st


st.title("Dianna's dashboard")

with st.sidebar:
    st.header('Input data')

    image = st.file_uploader('Image')

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
                st.number_input('Number of masks', value=1000)
                st.number_input('Feature resolution', value=6)
            with c2:
                st.number_input('Probability to be kept unmasked', value=0.1)

        if method == 'KernelSHAP':

            with c1:
                st.number_input('Number of samples', value=1000)
                st.number_input('Background', value=0)
            with c2:
                st.number_input('Number of segments', value=200)
                st.number_input('σ', value=0)

        if method == 'LIME':

            with c1:
                st.number_input('Random state', value=2)

c1, c2 = st.columns(2)

with c1:
    st.button('Update explanation', type='primary')

with c2:
    st.button('Stop explanation', type='secondary')
