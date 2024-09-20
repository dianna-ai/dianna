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
    st.image(str(data_directory / 'logo.png'), width = 360)

    st.markdown("""
    **DIANNA** (Deep Insight And Neural Network Analysis) is a Python package that brings explainable AI (XAI)
    to your research project. <br>
    It wraps _systematically_ selected XAI methods (**explainers**) in a simple, uniform interface.<br>
    It's built by, with and for academic researchers and research software engineers
    who use AI, but users not need to be XAI experts! <br>
    DIANNA supports the de-facto standard format of neural network models - [ONNX](https://onnx.ai/:).

    ### Dashboard
    The DIANNA dashboard can be used for explanation of the outcomes of several ONNX models trained for the tasks
    and datasets presented
    in the [DIANNA Tutorials](https://github.com/dianna-ai/dianna/tree/main/tutorials#datasets-and-tasks).

    The dashboard shows the visual explanation of a models' outcome
    on a selected data _instance_ by one or more selected explainers. <br>
    It allows you to compare the results of different explainers, as well as explanations
    of the top ranked predicted model outcomes.

    There are separate sections for each of the different _data modalities_ supported by DIANNA:
    :gray-background[**Image**], :gray-background[**Text**],
    :gray-background[**Tabular**], and :gray-background[**Time series**] data. <br>
    The visual explanation is an overlaid on the data instance :rainbow-background[**heatmap**]
    highlighting the relevance (attribution) of each data instance _element_ to a selected model's outcome.<br>
    The data element for images is a (super)pixel, for text - a word, for tabular data - an attribute,
    and for time-series -  a time interval. Attributions can be positive, negative or irrelevant.<br>
    To interpret heatmaps, note that attributions are bound between -1 and 1.
    The maximum (positive) value is set to 1 and the minimum (negative)  value to -1.<br>
    The dashboard uses the _bwr  (blue white red)_ colormap assigning :blue[**blue**] color to negative
    relevances, **white** color to near-zero values, and :red[**red**] color to positive values.

    """,
                unsafe_allow_html=True)

    st.image(str(data_directory / 'colormap.png'), width = 660)

    st.markdown("""
    The dashboard _primarily_ illustrates the examples from the DIANNA tutorials.

    It is also possible to upload _own_ trained (ONNX) model and data item for which you would like
    the model's decision explanation.<br>
    You can then select the explainer you want to use and set its hyperparameters.

     ### More information

    - [Source code](https://github.com/dianna-ai/dianna)
    - [Documentation](https://dianna.readthedocs.io/)
    - [XAI choice](https://blog.esciencecenter.nl/how-to-find-your-artificial-intelligence-explainer-dbb1ac608009)
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
