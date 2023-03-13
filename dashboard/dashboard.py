"""Start dashboard."""
import warnings
from importlib import reload
import dash
import layouts
import utilities
from callbacks import app
from dash import dcc
from dash import html
from layouts import images_page
from layouts import text_page


warnings.filterwarnings('ignore')  # disable warnings related to versions of tf

reload(layouts)
reload(utilities)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    """Load content of dashboard."""
    if pathname == '/apps/text':
        return text_page
    return images_page  # home page


if __name__ == '__main__':
    app.run_server(debug=True)
