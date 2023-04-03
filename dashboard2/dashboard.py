import streamlit as st


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

st.image(
    'https://user-images.githubusercontent.com/3244249/151994514-b584b984-a148-4ade-80ee-0f88b0aefa45.png'
)

st.title("Dianna's dashboard")

st.write("""
DIANNA is a Python package that brings explainable AI (XAI) to your research project.
It wraps carefully selected XAI methods in a simple, uniform interface. It's built by,
with and for (academic) researchers and research software engineers working on machine
learning projects.

- [Images](/images)
- [Text](/text)
""")
