"""Browser entrypoint: runs the dashboard like `dianna-dashboard` does, with Home.py as the main script.

Like the Streamlit server, a URL path naming a page (e.g. /Images) runs that page directly.
"""
import runpy
import sys
from urllib.parse import urlparse
import streamlit as st
from web_setup import DASHBOARD
from web_setup import register_components

if str(DASHBOARD) not in sys.path:
    sys.path.insert(0, str(
        DASHBOARD))  # the dashboard imports its helpers as top-level modules
register_components()

name = urlparse(st.context.url or '').path.rstrip('/').rsplit('/', 1)[-1]
page = DASHBOARD / 'pages' / f'{name}.py'
if name and page.exists():
    st.set_page_config(page_title=name.replace('_', ' '))
    runpy.run_path(str(page), run_name='__main__')
else:
    runpy.run_path(str(DASHBOARD / 'Home.py'), run_name='__main__')
