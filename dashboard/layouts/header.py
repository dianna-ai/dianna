import base64
from pathlib import Path
from dash import html
from .styles import COLORS


this_dir = Path(__file__).parents[1]

image_filename = this_dir / 'app_data' / 'logo.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())


def get_header():
    """Creates layout for header row."""
    header = html.Div([

        html.Div([],
            className='four columns',
            style = {'padding-top' : '1%'}
        ),

        html.Div([
            html.H1(children='DIANNA\'s Dashboard',
                    style = {'textAlign' : 'center', 'color' : COLORS['white']}
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
        style = {'background-color' : COLORS['blue1']}
        )

    return header
