import base64
import utilities
from dash import dcc
from dash import html


#static images
image_filename = 'app_data/logo.png' # replace with your own image
# pylint: disable=consider-using-with
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

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
    """Creates layout for header row."""
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
                    src = f'data:image/png;base64,{encoded_image.decode()}',
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
    """Creates layout for navbar row."""
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

    return navbar_text

# uploads images bar        
def get_uploads_images():
    """Creates layout for images page."""
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
                'height': '230px'
                }
            ), 
        
            # select model row
            html.Div([
                dcc.Upload(
                    id='upload-model-img',
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
            html.Div(id='output-model-img-upload',
            className = 'row',
            style = {
                'height': '230px',
                'margin-top': '20px',
                #'margin-bottom': '50px',
                #'fontSize': 8,
                'color' : colors['blue1']}
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
                html.Div([
                    dcc.Markdown(children='**Select one/more XAI methods**',
                        style = {'margin-top': '20px', 'textAlign' : 'center', 'color' : colors['blue1']}
                        ),
                    dcc.Dropdown(id = 'method_sel_img',
                        options = [{'label': 'RISE', 'value': 'RISE'},
                                {'label': 'KernelSHAP', 'value': 'KernelSHAP'},
                                {'label': 'LIME', 'value': 'LIME'}],
                        placeholder = "Select methods",
                        value=[""],
                        multi = True,
                        style={
                                'margin-left': '0px',
                                'margin-top': '10px',
                                'width': '100%',
                                'color' : colors['blue1']
                            }
                    )
                ], className = 'six columns'
                ),
                html.Div([
                    dcc.Markdown(children='**Number of top results to show**',
                        style = {'margin-top': '20px', 'textAlign' : 'center', 'color' : colors['blue1']}
                        ),
                    dcc.Input(id = 'show_top',
                            type="number",
                            value=2,
                            style={
                                    'margin-right': '10px',
                                    'margin-top': '2px',
                                    'width': '100%',
                                    'color' : colors['blue1']
                                }
                        )
                ], className = 'three columns'
                ),
            ],
            className = 'row', style = {'padding-bottom' : '3%'}
            ),
            # Settings bar
            html.Div([
                html.Div([
                    html.H6(children='XAI method specific settings', style={'font-weight': 'bold'}),
                ], className='nine columns'
                ),
            ],
            className = 'row'
            ),
            # XAI method settings buttons
            html.Div([
                html.Div([
                    dcc.Markdown(children='**Rise**',
                        style = {'margin-top': '20px', 'textAlign' : 'center', 'color' : colors['blue1']}
                        ),
                    dcc.Markdown(children='Number of masks',
                        style = {'margin-left': '5px', 'margin-top': '0px', 'textAlign' : 'left', 'color' : colors['blue1']}
                        ),
                    dcc.Input(id = 'n_masks',
                        placeholder = "Number of masks",
                        type="number",
                        value=1000,
                        style={
                                'margin-right': '20px',
                                'margin-top': '0px',
                                'width': '100%',
                                'color' : colors['blue1']
                            }
                    ),
                    dcc.Markdown(children='Feature resolution',
                        style = {'margin-left': '5px', 'margin-top': '5px', 'textAlign' : 'left', 'color' : colors['blue1']}
                        ),
                    dcc.Input(id = 'feature_res',
                        placeholder = "Feature res",
                        type="number",
                        value=6,
                        style={
                                'margin-right': '20px',
                                'margin-top': '0px',
                                'width': '100%',
                                'color' : colors['blue1']
                            }
                    ),
                    dcc.Markdown(children='Probability to be kept unmasked',
                        style = {'margin-left': '5px', 'margin-top': '5px', 'textAlign' : 'left', 'color' : colors['blue1']}
                        ),
                    dcc.Input(id = 'p_keep',
                        placeholder = "P keep",
                        type="number",
                        value=0.1,
                        style={
                                'margin-right': '20px',
                                'margin-top': '0px',
                                'width': '100%',
                                'color' : colors['blue1']
                            }
                    )
                ], className = 'three columns'),
                html.Div([
                    dcc.Markdown(children='**KernelShap**',
                        style = {'margin-top': '20px', 'textAlign' : 'center', 'color' : colors['blue1']}
                        ),
                    dcc.Markdown(children='Number of samples',
                        style = {'margin-left': '5px', 'margin-top': '0px', 'textAlign' : 'left', 'color' : colors['blue1']}
                        ),
                    dcc.Input(id = 'n_samples',
                        placeholder = "N samples",
                        type="number",
                        value=1000,
                        style={
                                'margin-right': '20px',
                                'margin-top': '0px',
                                'width': '100%',
                                'color' : colors['blue1']
                            }
                    ),
                    dcc.Markdown(children='Background',
                        style = {'margin-left': '5px', 'margin-top': '5px', 'textAlign' : 'left', 'color' : colors['blue1']}
                        ),
                    dcc.Input(id = 'background',
                        placeholder = "Background",
                        type="number",
                        value=0,
                        style={
                                'margin-right': '20px',
                                'margin-top': '0px',
                                'width': '100%',
                                'color' : colors['blue1']
                            }
                    ),
                    dcc.Markdown(children='Number of segments',
                        style = {'margin-left': '5px', 'margin-top': '5px', 'textAlign' : 'left', 'color' : colors['blue1']}
                        ),
                    dcc.Input(id = 'n_segments',
                        placeholder = "N segments",
                        type="number",
                        value=200,
                        style={
                                'margin-right': '20px',
                                'margin-top': '0px',
                                'width': '100%',
                                'color' : colors['blue1']
                            }
                    ),
                    dcc.Markdown(children='$$\sigma$$', mathjax=True, 
                        style = {'margin-left': '5px', 'margin-top': '5px', 'textAlign' : 'left', 'color' : colors['blue1']}
                        ),
                    dcc.Input(id = 'sigma',
                        placeholder = "sigma",
                        type="number",
                        value=0,
                        style={
                                'margin-right': '20px',
                                'margin-top': '0px',
                                'width': '100%',
                                'color' : colors['blue1']
                            }
                    )
                ], className = 'three columns'),
                html.Div([
                    dcc.Markdown(children='**Lime**',
                        style = {'margin-top': '20px', 'textAlign' : 'center', 'color' : colors['blue1']}
                        ),
                    dcc.Markdown(children='Random state',
                        style = {'margin-left': '5px', 'margin-top': '5px', 'textAlign' : 'left', 'color' : colors['blue1']}
                        ),
                    dcc.Input(id = 'random_state',
                        placeholder = "Random state",
                        type="number",
                        value=2,
                        style={
                                'margin-right': '20px',
                                'margin-top': '0px',
                                'width': '100%',
                                'color' : colors['blue1']
                            }
                    )
                ], className = 'three columns')
            ],
            className = 'row'
            ),

            # printing predictions
            html.Div(
                id='output-state-img',
                className = 'row'),

            html.Div([
                dcc.Loading(children=[
                    dcc.Graph(
                        id='graph_img',
                        figure = utilities.blank_fig())],
                    color=colors['blue1'], type="dot", fullscreen=False)],
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


# uploads images bar        
def get_uploads_text():
    """Creates layout for text page."""
    # uploads row
    uploads = html.Div([

        # uploads first col (3-col)
        html.Div([

            # insert text row
            html.Div([
                dcc.Input(
                    id='upload-text',
                    placeholder='Type here input string...',
                    value='',
                    type='text',
                    style={
                        'width': '80%',
                        'height': '40px',
                        'lineHeight': '40px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '3px',
                        'margin-left': '30px',
                        'margin-top': '20px',
                        'textAlign': 'center',
                        'align-items': 'center',
                        'color' : colors['blue1']
                    },
                ),
            ],
            className = 'row',
            ),

            # submit text row
            html.Div([
                html.Button(
                    id='submit-text',
                    type='submit',
                    children='Submit',
                    style={
                        'background-color' : colors['blue2'],
                        'color' : colors['white']
                    })
            ],
            className = 'row',
            style={
                'borderWidth': '1px',
                'margin-top': '10px',
                'textAlign': 'center',
                'align-items': 'center'
                }
            ),

            # plot inserted text row
            html.Div(id='text_test',
            className = 'row',
            style = {
                'height': '130px',
                'margin-top': '20px',
                }
            ), 
        
            # select model row
            html.Div([
                dcc.Upload(
                    id='upload-model-text',
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
            html.Div(id='output-model-text-upload',
            className = 'row',
            style = {
                'height': '230px',
                'margin-top': '20px',
                'color' : colors['blue1']}
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
                dcc.Dropdown(id = 'method_sel_text',
                    options = [{'label': 'RISE', 'value': 'RISE'},
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
                id='output-state-text',
                className = 'row',
                style = {
                    'margin-top': '20px',
                    'color' : colors['blue1']
                    }
                    ),

            # plotting explainations
            html.Div([
                dcc.Graph(
                    id='graph_text_lime',
                    figure = utilities.blank_fig())],
                    className = 'row',
                    style = {
                        'margin-top': '80',
                        'margin-left': '140px',
                        'height': '100px'
                    }),

            # plotting explainations
            html.Div([
                dcc.Graph(
                    id='graph_text_rise',
                    figure = utilities.blank_fig())],
                    className = 'row',
                    style = {
                        'margin-top': '80',
                        'margin-left': '140px',
                        'height': '100px'
                    })

        ], 
        className = 'nine columns')

    ], className = 'row',
    style = { 
        'background-color' : colors['blue4'],
        'textAlign': 'center',
        'align-items': 'center',
        'height': '600px'
        })

    return uploads


images_page = html.Div([
    
    #Row 1 : Header
    get_header(),
    #Row 2 : Nav bar
    get_navbar("images"),

    get_uploads_images(),

    # hidden signal value
    dcc.Store(id='signal_image'),
    
    ])

text_page = html.Div([
    
    #Row 1 : Header
    get_header(),
    #Row 2 : Nav bar
    get_navbar("text"),

    get_uploads_text(),

    # hidden signal value
    dcc.Store(id='signal_text'),
    
    ])
