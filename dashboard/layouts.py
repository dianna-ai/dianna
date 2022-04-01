import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from jupyter_dash import JupyterDash
from dash import html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import base64
import utilities

#static images
image_filename = 'app_data/logo.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

#rise = base64.b64encode(open('rise.png', 'rb').read())
#kernels = base64.b64encode(open('kernels.png', 'rb').read())

# colors
colors = {
    'white': '#FFFFFF',
    'text': '#091D58',
    'blue1' : '#063446', #dark blue
    'blue2' : '#0e749b',
    'blue3' : '#15b3f0',
    'blue4' : '#E4F3F9', #light blue
    'yellow1' : '#f0d515'
}

# styles
navbarcurrentpage = {
    'text-decoration' : 'underline',
    'text-decoration-color' : colors['yellow1'],
    'color' : colors['white'],
    'text-shadow': '0px 0px 1px rgb(251, 251, 252)',
    'textAlign' : 'center'
    }

navbarotherpage = {
    'text-decoration' : 'underline',
    'text-decoration-color' : colors['blue2'],
    'color' : colors['white'],
    'textAlign' : 'center'
    }

# app layout
# In Bootstrap, the "row" class is used mainly to hold columns in it.
# Bootstrap divides each row into a grid of 12 virtual columns.

# header
def get_header():

    header = html.Div([

        html.Div([],
            className='four columns',
            style = {'padding-top' : '1%'}
        ),

        html.Div([
            html.H1(children='DIANNA\'s Dashboard',
                    style = {'textAlign' : 'center', 'color' : colors['white']}
            )],
            className='four columns',
            style = {'padding-top' : '1%'}
        ),

        html.Div([
            html.Img(
                    src = 'data:image/png;base64,{}'.format(encoded_image.decode()),
                    height = '43 px',
                    width = 'auto')
            ],
            className = 'four columns',
            style = {
                    'textAlign': 'right',
                    'padding-top' : '1.3%',
                    'padding-right' : '4%',
                    'height' : 'auto'
                    })

        ],
        className = 'row',
        style = {'background-color' : colors['blue1']}
        )

    return header

# nav bar
def get_navbar(p = 'images'):

    navbar_images = html.Div([

        html.Div(['b'],
            className = 'five columns',
            style = {'color' : colors['blue2']}
        ),

        html.Div([
            dcc.Link(
                html.H4(children = 'Images',
                        style = navbarcurrentpage),
                href='/apps/images'
                )
        ],
        className='one column'),

        html.Div([
            dcc.Link(
                html.H4(children = 'Text',
                    style = navbarotherpage),
                href='/apps/text'
                )
        ],
        className='one column'),

        html.Div([], className = 'five columns')

    ],
    
    className = 'row',
    style = {'background-color' : colors['blue2']
            }
    )

    navbar_text = html.Div([

        html.Div(['b'],
            className = 'five columns',
            style = {'color' : colors['blue2']}
        ),

        html.Div([
            dcc.Link(
                html.H4(children = 'Images',
                    style = navbarotherpage),
                href='/apps/images'
                )
        ],
        className='one column'),

        html.Div([
            dcc.Link(
                html.H4(children = 'Text',
                    style = navbarcurrentpage),
                href='/apps/text'
                )
        ],
        className='one column'),

        html.Div([], className = 'five columns')

    ],
    
    className = 'row',
    style = {'background-color' : colors['blue2']
            }
    )

    if p == 'images':
        return navbar_images
    else:
        return navbar_text

# uploads bar        
def get_uploads():

    # uploads row
    uploads = html.Div([

        # uploads first col (3-col)
        html.Div([

            # select image row
            html.Div([
                dcc.Upload(
                    id='upload-image',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Image')
                    ]),
                    style={
                        'width': '80%',
                        'height': '40px',
                        'lineHeight': '40px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '3px',
                        'margin-left': '30px',
                        'margin-top': '20px',
                        'color' : colors['blue1']
                    },
                    # Allow multiple files to be uploaded
                    multiple=True
                )
            ],
            className = 'row',
            ),

            # plot selected image row
            html.Div(dcc.Graph(
                    id='graph_test',
                    figure = utilities.blank_fig()
                    ),
            className = 'row',
            style = {
                #'height': '230px'
                }
            ), 
        
            # select model row
            html.Div([
                dcc.Upload(
                    id='upload-model',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Model')
                    ]),
                    style={
                        'width': '80%',
                        'height': '40px',
                        'lineHeight': '40px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '3px',
                        'margin-left': '30px',
                        'margin-top': '20px',
                        'color' : colors['blue1']
                    },
                    # Allow multiple files to be uploaded
                    multiple=True
                ),
            ],
            className = 'row', 
            ),

            # print selected model row
            html.Div(id='output-model-upload',
            className = 'row',
            style = {
                'height': '230px'}
            )

            ],
            className = 'three columns',
            style = {
                'textAlign': 'center',
                'align-items': 'center'
            }),

        # XAI methods col (9-col)
        html.Div([

            # XAI methods selection row
            html.Div([
                dcc.Dropdown(id = 'method_sel',
                    options = [{'label': 'RISE', 'value': 'RISE'},
                            {'label': 'KernelSHAP', 'value': 'KernelSHAP'},
                            {'label': 'LIME', 'value': 'LIME'}],
                    placeholder = "Select one/more XAI methods",
                    value=[""],
                    multi = True,
                    style={
                            'margin-left': '155px',
                            'margin-top': '20px',
                            'width': '60%',
                            'color' : colors['blue1']
                        }
                )
            ],
            className = 'row'
            ),

            # printing predictions
            html.Div(
                id='output-state',
                className = 'row'),

            html.Div([
                dcc.Graph(
                    id='graph',
                    figure = utilities.blank_fig())],
                    className = 'row',
                    style = {
                        'margin-left': '140px',
                    })

        ], 
        className = 'nine columns')

    ], className = 'row',
    style = { 
        'background-color' : colors['blue4'],
        'textAlign': 'center',
        'align-items': 'center'
        })

    return uploads

# def get_method_sel():

#     method_sel = html.Div([

#         html.Div(['b'],
#             className = 'five columns',
#             style = {'color' : colors['blue4']}),
#         html.Div([
#             dcc.Dropdown(id = 'method_sel2',
#                 options = [{'label': 'RISE', 'value': 'RISE'},
#                            {'label': 'KernelSHAP', 'value': 'KernelSHAP'},
#                            {'label': 'LIME', 'value': 'LIME'}],
#                 placeholder = "Select one/more methods",
#                 value=[],
#                 multi = True
#             ),
            #html.Button(
            #    id='submit-val',
            #    children = html.Div(['Get explanation']), n_clicks=0,
            #    style={
            #        'width': '100%',
            #        #'height': '40px',
            #        #'lineHeight': '40px',
            #        'borderWidth': '1px',
            #        #'borderStyle': 'dashed',
            #        'borderRadius': '3px',
            #        'textAlign': 'center',
            #        'align-items': 'center',
            #        'margin': '10px',
            #        'color' : colors['white'],
            #        'background-color' : colors['blue2']}
            #    )
            # ], 
    #         className = 'two columns')
    # ], className = 'row', style = {
    #     'textAlign': 'center',
    #     'background-color' : colors['blue4'],
    #     'align-items': 'center'})

    # return method_sel

# def get_pred():

#     pred = html.Div([
#         html.Div(['b'],
#             className = 'five columns',
#             style = {'color' : colors['blue4']}),
#         html.Div(id='output-state', className = 'two columns')
#         ], className = 'row', style = {
#         'textAlign': 'center',
#         'background-color' : colors['blue4'],
#         'align-items': 'center'})

#     return pred

# def get_method_plot():

#     method_plot = html.Div([
#         html.Div(['b'],
#             className = 'five columns',
#             style = {'color' : colors['blue4']}),
#         html.Div([dcc.Graph(id='rise', figure = utilities.blank_fig())], className = 'two columns'),
#         ],
#         className="row",
#         style={
#             'background-color' : colors['blue4'],
#             'textAlign': 'center',
#             'align-items': 'center',
#             'verical-align': 'center'})

#     return method_plot