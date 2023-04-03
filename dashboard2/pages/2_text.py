import streamlit as st


st.title("Dianna's dashboard")

with st.sidebar:
    st.header('Input data')

    text = st.text_input('Input string')

    model = st.file_uploader('Model')

    labels = st.file_uploader('Labels')

methods = st.multiselect('Select XAI methods', options=('Rise', 'KernelSHAP'))

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

        if method == 'LIME':

            with c1:
                st.number_input('Random state', value=2)

c1, c2 = st.columns(2)

with c1:
    st.button('Update explanation', type='primary')

with c2:
    st.button('Stop explanation', type='secondary')
