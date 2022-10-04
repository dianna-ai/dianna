# Dash&Flask
import dash
from dash import html, dcc
# Custom libraries
from dianna import utils
import layouts
from layouts import images_page, text_page
import utilities
import callbacks
from callbacks import app
# Others
import warnings
warnings.filterwarnings('ignore') # disable warnings relateds to versions of tf

# debug
from importlib import reload
reload(layouts)
reload(utilities)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/text':
         return text_page
    else:
         return images_page # home page


if __name__ == '__main__':
    app.run_server(debug=True)