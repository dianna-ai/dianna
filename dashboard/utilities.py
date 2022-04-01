import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from jupyter_dash import JupyterDash
from dash import html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import base64
import numpy as np

colors = {
    'white': '#FFFFFF',
    'text': '#091D58',
    'blue1' : '#063446',
    'blue2' : '#0e749b',
    'blue3' : '#15b3f0',
    'blue4' : '#E4F3F9',
    'yellow1' : '#f0d515'
}

def blank_fig():
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=colors['blue4'],
        plot_bgcolor = colors['blue4'])
    fig.update_xaxes(gridcolor = colors['blue4'], showticklabels = False, zerolinecolor=colors['blue4'])
    fig.update_yaxes(gridcolor = colors['blue4'], showticklabels = False, zerolinecolor=colors['blue4'])
    
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

# For KernelSHAP: fill each pixel with SHAP values
def fill_segmentation(values, segmentation):
    out = np.zeros(segmentation.shape)
    for i in range(len(values)):
        out[segmentation == i] = values[i]
    return out

# For LIME: we divided the input data by 256 for the models and LIME needs RGB values
def preprocess_function(image):
    return (image / 256).astype(np.float32)