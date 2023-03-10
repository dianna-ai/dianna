from dash import dcc
from dash import html
from .styles import COLORS


navbarcurrentpage = {
        'text-decoration' : 'underline',
        'text-decoration-color' : COLORS['yellow1'],
        'color' : COLORS['white'],
        'text-shadow': '0px 0px 1px rgb(251, 251, 252)',
        'textAlign' : 'center'
}
    
navbarotherpage = {
    'text-decoration' : 'underline',
    'text-decoration-color' : COLORS['blue2'],
    'color' : COLORS['white'],
    'textAlign' : 'center'
}


def get_navbar(p="images"):
    """Creates layout for navbar row."""
    navbar_images = html.Div([

        html.Div(['b'],
            className = 'five columns',
            style = {'color' : COLORS['blue2']}
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
    style = {'background-color' : COLORS['blue2']
            }
    )

    navbar_text = html.Div([

        html.Div(['b'],
            className = 'five columns',
            style = {'color' : COLORS['blue2']}
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
    style = {'background-color' : COLORS['blue2']
            }
    )

    if p == 'images':
        return navbar_images

    return navbar_text