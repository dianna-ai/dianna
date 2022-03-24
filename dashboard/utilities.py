import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from jupyter_dash import JupyterDash
from dash import html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import base64

colors = {
    'white': '#FFFFFF',
    'text': '#091D58',
    'blue1' : '#063446',
    'blue2' : '#0e749b',
    'blue3' : '#15b3f0',
    'blue4' : '#d0f0fc',
    'yellow1' : '#f0d515'
}

def blank_fig():
    fig = go.Figure(go.Scatter(x=[], y = []))
    fig.update_layout(template = None, paper_bgcolor=colors['blue4'])
    fig.update_xaxes(showgrid = False, showticklabels = False, zeroline=False)
    fig.update_yaxes(showgrid = False, showticklabels = False, zeroline=False)
    
    return fig

def parse_contents_image(contents, filename):
    return html.Div([
        html.H5(filename + ' loaded'),
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents, height = '160 px', width = 'auto')
    ])

def parse_contents_model(contents, filename):
    return html.Div([
        html.H5(filename + ' loaded')
    ])