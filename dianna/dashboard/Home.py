import importlib
import streamlit as st
from _shared import add_sidebar_logo
from _shared import data_directory
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Dianna's dashboard",
                   page_icon='📊',
                   layout='centered',
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
    "Text": "pages.Text",
    "Time series": "pages.Time_series"
}

# Set up the top menu
selected = option_menu(
    menu_title=None,
    options=["Home", "Images", "Text", "Time series"],
    icons=["house", "camera", "alphabet", "clock"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

# Display the content of the selected page
if selected == "Home":
    add_sidebar_logo()

    st.image(str(data_directory / 'logo.png'))

    st.markdown("""
    DIANNA is a Python package that brings explainable AI (XAI) to your research project.
    It wraps carefully selected XAI methods in a simple, uniform interface. It's built by,
    with and for (academic) researchers and research software engineers working on machine
    learning projects.

    ### Pages

    - <a href="/Images" target="_parent">Images</a>
    - <a href="/Text" target="_parent">Text</a>
    - <a href="/Time_series" target="_parent">Time series</a>


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
    if selected != 'Text':
        for k in st.session_state.keys():
            if 'Text' in k:
                st.session_state.pop(k, None)
    if selected != 'Time series':
        for k in st.session_state.keys():
            if 'TS' in k:
                st.session_state.pop(k, None)
    page = importlib.import_module(page_module)
