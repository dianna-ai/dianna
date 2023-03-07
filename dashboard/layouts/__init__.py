"""Layout submodule."""
import utilities
from dash import dcc
from dash import html
from .header import get_header
from .images import get_uploads_images
from .navbar import get_navbar
from .text import get_uploads_text


images_page = html.Div([
    get_header(),
    get_navbar(p="images"),
    get_uploads_images(),

    # hidden signal value
    dcc.Store(id='signal_image'),
    
    ])

text_page = html.Div([
    
    get_header(),
    get_navbar(p="text"),
    get_uploads_text(),

    # hidden signal value
    dcc.Store(id='signal_text'),
    
    ])
