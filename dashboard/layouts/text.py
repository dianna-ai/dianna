import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from .fig import blank_fig
from .styles import COLORS, astyle


def get_uploads_text():
    """Creates layout for text page."""
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
                        'color' : COLORS['blue1']
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
                        'background-color' : COLORS['blue2'],
                        'color' : COLORS['white']
                    })
            ],
            className = 'row',
            style={
                'borderWidth': '1px',
                'margin': '10px',
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
                        html.A('Select Model', style=astyle)
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
                        'color' : COLORS['blue1']
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
                'height': '100px',
                'margin-top': '20px',
                'color' : COLORS['blue1']}
            ),

            # select label row
            html.Div([
                dcc.Upload(
                    id='upload-label-text',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Label File', style=astyle)
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
                        'color' : COLORS['blue1']
                    },
                    # Allow multiple files to be uploaded
                    multiple=False
                ),
            ],
            className = 'row', 
            ),

            # print selected model row
            html.Div(id='output-label-text-upload',
            className = 'row',
            style = {
                'height': '100px',
                'margin-top': '20px',
                #'margin-bottom': '50px',
                #'fontSize': 8,
                'color' : COLORS['blue1']}
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
                        style = {'margin-top': '20px', "margin-left": "50px",
                            'textAlign' : 'center',
                            'color' : COLORS['blue1']}
                        ),
                    dcc.Checklist(id = 'method_sel_text',
                        options = [{'label': 'RISE', 'value': 'RISE'},
                                {'label': 'LIME', 'value': 'LIME'}],
                        inline=True,
                        inputStyle={"margin-left": "50px", "margin-right": "5px"},
                        style={
                                'margin-left': '0px',
                                'margin-top': '10px',
                                'textAlign' : 'center',
                                'width': '100%',
                                'color' : COLORS['blue1']
                            }
                    )
                ], className = 'six columns'
                ),
            ],
            className = 'row', style = {'padding-bottom' : '1%'}
            ),
            html.Div([
                # update button
                html.Button('Update explanation',
                        id='update_button_t',
                        n_clicks=0,
                        style={
                            'margin-left': '0px',
                            'margin-top': '0px',
                            'width': '20%',
                            'float': 'left',
                            'backgroundColor': COLORS['blue2'],
                            'color' : COLORS['white']
                        }
                    ),
                html.Button('Stop Explanation',
                        id='stop_button_t',
                        n_clicks=0,
                        style={
                            'margin-left': '40px',
                            'margin-top': '0px',
                            'width': '20%',
                            'float': 'left',
                            'backgroundColor': COLORS['red1'],
                            'color' : COLORS['white']
                        }
                    ),
                ],
            className = 'row', style = {'padding-bottom' : '1%'}
            ),
            # Settings bar
            html.Div([
                html.Button(
                        "Click to show XAI method specific settings",
                        id="collapse-parameters-button",
                        n_clicks=0,
                        style={
                            'margin-left': '0px',
                            'margin-top': '0px',
                            'width': '40%',
                            'float': 'center',
                            'backgroundColor': COLORS['blue1'],
                            'color' : COLORS['white']
                        }
                    ),
                ],
            className = 'row'
            ),
            # XAI method settings buttons
            dbc.Collapse(
                html.Div([
                    html.Div([
                        dcc.Markdown(children='**Rise**',
                            style = {'margin-top': '20px', 'textAlign' : 'center',
                                'color' : COLORS['blue1']}
                            ),
                        dcc.Markdown(children='Number of masks',
                            style = {'margin-left': '5px', 'margin-top': '0px',
                                'textAlign' : 'left', 'color' : COLORS['blue1']}
                            ),
                        dcc.Input(id = 'n_masks_text',
                            placeholder = "Number of masks",
                            type="number",
                            value=1000,
                            style={
                                    'margin-right': '20px',
                                    'margin-top': '0px',
                                    'width': '100%',
                                    'color' : COLORS['blue1']
                                }
                        ),
                        dcc.Markdown(children='Feature resolution',
                            style = {'margin-left': '5px', 'margin-top': '5px',
                                'textAlign' : 'left', 'color' : COLORS['blue1']}
                            ),
                        dcc.Input(id = 'feature_res_text',
                            placeholder = "Feature res",
                            type="number",
                            value=6,
                            style={
                                    'margin-right': '20px',
                                    'margin-top': '0px',
                                    'width': '100%',
                                    'color' : COLORS['blue1']
                                }
                        ),
                        dcc.Markdown(children='Probability to be kept unmasked',
                            style = {'margin-left': '5px', 'margin-top': '5px',
                                'textAlign' : 'left', 'color' : COLORS['blue1']}
                            ),
                        dcc.Input(id = 'p_keep_text',
                            placeholder = "P keep",
                            type="number",
                            value=0.1,
                            style={
                                    'margin-right': '20px',
                                    'margin-top': '0px',
                                    'width': '100%',
                                    'color' : COLORS['blue1']
                                }
                        )
                    ], className = 'three columns'),
                    html.Div([
                        dcc.Markdown(children='**Lime**',
                            style = {'margin-top': '20px', 'textAlign' : 'center',
                                'color' : COLORS['blue1']}
                            ),
                        dcc.Markdown(children='Random state',
                            style = {'margin-left': '5px', 'margin-top': '5px',
                                'textAlign' : 'left', 'color' : COLORS['blue1']}
                            ),
                        dcc.Input(id = 'random_state_text',
                            placeholder = "Random state",
                            type="number",
                            value=2,
                            style={
                                    'margin-right': '20px',
                                    'margin-top': '0px',
                                    'width': '100%',
                                    'color' : COLORS['blue1']
                                }
                        )
                    ], className = 'three columns')
                ],
                className = 'row'
                ),
                id="collapse-parameters",
                is_open=False,
            ),

            # printing predictions
            html.Div(
                id='output-state-text',
                className = 'row',
                style = {
                    'margin-top': '20px',
                    'color' : COLORS['blue1']
                    }
                    ),

            # plotting explanations
            html.Div([
                dcc.Graph(
                    id='graph_text_rise',
                    figure = blank_fig())],
                    className = 'row',
                    style = {
                        'margin-top': '80',
                        'margin-left': '140px',
                        #'height': '100px'
                    }),

            # plotting explanations
            html.Div([
                dcc.Graph(
                    id='graph_text_lime',
                    figure = blank_fig())],
                    className = 'row',
                    style = {
                        'margin-top': '80',
                        'margin-left': '140px',
                        #'height': '100px'
                    })

        ], 
        className = 'nine columns')

    ], className = 'row',
    style = { 
        'background-color' : COLORS['blue4'],
        'textAlign': 'center',
        'align-items': 'center'
        })

    return uploads