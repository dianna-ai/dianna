import plotly.graph_objects as go
from dash import html
import numpy as np
import layouts
from PIL import Image, ImageStat

def blank_fig(text=None):
    fig = go.Figure(data=go.Scatter(x=[], y=[]))
    fig.update_layout(
        paper_bgcolor=layouts.colors['blue4'],
        plot_bgcolor = layouts.colors['blue4'])

    fig.update_xaxes(showgrid = False, showticklabels = False, zeroline=False)
    fig.update_yaxes(showgrid = False, showticklabels = False, zeroline=False)

    if text is not None:
        fig.update_layout(
            width=300,
            height=300,
            annotations = [
                        {   
                            "text": text,
                            "xref": "paper",
                            "yref": "paper",
                            "showarrow": False,
                            "font": {
                                "size": 14,
                                "color" : layouts.colors['blue1']
                            },
                            "valign": "top",
                            "yanchor": "top",
                            "xanchor": "center",
                            "yshift": 60,
                            "xshift": 10
                        }
                    ]
            )

    return fig

def open_image(path):
    im = Image.open(path).convert("RGB")
    stat = ImageStat.Stat(im)
    im = np.asarray(im).astype(np.float32)

    if sum(stat.sum)/3 == stat.sum[0]: #check the avg with any element value
        return np.expand_dims(im[:,:,0], axis=2) / 255 #if grayscale
    else:
        return im #else its colour

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