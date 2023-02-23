import pandas as pd

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from contents.app import *


@app.callback(
    # Table and Data
    Output('the-data', 'data'),
    Output('datatable', 'data'),
    Output('datatable', 'columns'),
    # Dropdowns
    Output('type-casting-feature-dropdown', 'options'),
    Output('drop-features-dropdown', 'options'),
    Output('transformations-base-features', 'options'),
    # Inputs
    Input('drop-features-button', 'n_clicks'),
    State('drop-features-dropdown', 'value'),
    State('the-data', 'data'),
)
def drop_features(n_clicks, features, data):
    df = pd.DataFrame.from_dict(data)
    if n_clicks and features and data:
        df = df.drop(features, axis=1)
        data = df.to_dict('records')
        features_names = [{'name': i, 'id': i} for i in df.columns],
        return [
            data,
            data,
            features_names,
            features_names,
            features_names,
            features_names,
        ]
    raise PreventUpdate

