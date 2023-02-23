import os
import pandas as pd

from dash import html, dcc
from dash_extensions.enrich import DashProxy, MultiplexerTransform
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from dash import dash_table
import dash_bootstrap_components as dbc
import dash_loading_spinners as dls
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import plotly.graph_objs as go

dbc_css = ("https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.2/dbc.min.css")
external_stylesheets = [dbc.themes.DARKLY, dbc_css]
app = DashProxy(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,
    prevent_initial_callbacks=True,
    transforms=[MultiplexerTransform()],
)

app.layout = html.Div([
    html.Div(id='a', children='a'),
    html.Br(),
    dbc.Button(
        'Button',
        id='button',
        outline=True,
    ),
    dbc.Popover(
        [],
        target='a',
        trigger='hover',
        hide_arrow=True,
        id='popover',
    ),
])

@app.callback(
    Output('button', 'outline'),
    Input('popover', 'is_open'),
)
def update_b(is_open):
    if is_open:
        return False
    return True

if __name__ == '__main__':
    app.run_server(debug=True)
