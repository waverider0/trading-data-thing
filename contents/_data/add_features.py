import pandas as pd

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from contents.app import *


@app.callback(
    Output('add-features-table', 'data'),
    Output('add-features-table', 'columns'),
    Input('add-features-refresh', 'n_clicks'),
    State('the-data', 'data')
)
def render_table(n_clicks, data):
    if n_clicks and data:
        df = pd.DataFrame.from_dict(data)
        return [
            data,
            [{'name': i, 'id': i} for i in df.columns],
        ]
    raise PreventUpdate