import pandas as pd

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from contents.app import *

@app.callback(
    Output('preprocessing-table', 'data'),
    Output('preprocessing-table', 'columns'),
    Input('preprocessing-refresh', 'n_clicks'),
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