import importlib
import streamlit as st
from _shared import data_directory
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Dianna's dashboard",
                   page_icon='📊',
                   layout='wide',
                   initial_sidebar_state='auto',
                   menu_items={
                       'Get help':
                       'https://dianna.readthedocs.org',
                       'Report a bug':
                       'https://github.com/dianna-ai/dianna/issues',
                       'About':
                       ("Dianna's dashboard. Created by the Dianna team: "
                        'https://github.com/dianna-ai/dianna')
                   })

# Define dictionary of dashboard pages
pages = {
    "Home": "home",
    "Images": "pages.Images",
    "Tabular": "pages.Tabular",
    "Text": "pages.Text",
    "Time series": "pages.Time_series"
}

# Set up the top menu
selected = option_menu(
    menu_title=None,
    options=list(pages.keys()),
    icons=["house", "camera", "table", "alphabet", "clock"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

# Display the content of the selected page
if selected == "Home":
    st.image(str(data_directory / 'logo.png'))

    st.markdown("""
    DIANNA (Deep Insight And Neural Network Analysis) is a Python package that brings explainable AI (XAI) to your research project.
    It wraps carefully selected XAI methods in a simple, uniform interface. It's built by,
    with and for (academic) researchers and research software engineers working on machine
    learning projects. DIANNA supports the de-facto standard of neural network models - ONNX.
    
    ### Dashboard
    The DIANNA dashboard can be used for explanation of the behaviour of several ONNX models trained
    for the tasks and datasets presented in the
    [DIANNA Tutorials](https://github.com/dianna-ai/dianna/tree/main/tutorials#datasets-and-tasks ).
    The dashboard shows the visual explanation of a models' decision on a selected data item by a
    selected XAI method (explainer). It allows you to compare the results of different explainers,
    as well as explanations of the top ranked predicted labels. The dashboard was created using
    [streamlit](https://streamlit.io/).
    
    This dashboard provides pages for the different data items for which DIANNA can be used; 
    Image data, Tabular data, Text data, and Time series data.
    Here you can upload a trained (onnx) model, data item and the other data required for the
    specific data item. You can then select the explainer you want to use and set the different
    explainer variables. It is also possible to load example data items on each page.

    ### More information

    - [Source code](https://github.com/dianna-ai/dianna)
    - [Documentation](https://dianna.readthedocs.io/)
    """,
                unsafe_allow_html=True)

else:
    # Dynamically import and execute the page
    page_module = pages[selected]
    # Make sure that all variables are reset when switching page
    if selected != 'Images':
        for k in st.session_state.keys():
            if 'Image' in k:
                st.session_state.pop(k, None)
    if selected != 'Tabular':
        for k in st.session_state.keys():
            if 'Tabular' in k:
                st.session_state.pop(k, None)
    if selected != 'Text':
        for k in st.session_state.keys():
            if 'Text' in k:
                st.session_state.pop(k, None)
    if selected != 'Time series':
        for k in st.session_state.keys():
            if 'TS' in k:
                st.session_state.pop(k, None)
    page = importlib.import_module(page_module)
