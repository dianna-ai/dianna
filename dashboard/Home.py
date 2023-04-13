import streamlit as st
from _model_utils import data_directory


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

st.image(str(data_directory / 'logo.png'))

st.markdown("""
DIANNA is a Python package that brings explainable AI (XAI) to your research project.
It wraps carefully selected XAI methods in a simple, uniform interface. It's built by,
with and for (academic) researchers and research software engineers working on machine
learning projects.

### Pages

- <a href="/Text" target="_parent">Text</a>
- <a href="/Images" target="_parent">Images</a>

### More information

- [Source code](https://github.com/dianna-ai/dianna)
- [Documentation](https://dianna.readthedocs.io/)
""",
            unsafe_allow_html=True)
