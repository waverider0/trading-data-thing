import pandas as pd

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from contents.app import *


@app.callback(
    Output('the-data', 'data'),
    Output('hidden-div', 'children'),
    Input('remove-nulls-button', 'n_clicks'),
    State('the-data', 'data')
)
def remove_nulls(n_clicks, data):
    if n_clicks:
        df = pd.DataFrame.from_dict(data).dropna()
        return df.to_dict('records'), ''
    raise PreventUpdate