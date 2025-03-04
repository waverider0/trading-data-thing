import io
import base64
import numpy as np
import pandas as pd

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from contents.app import *

@app.callback(
    Output('the-data', 'data'),
    Output('hidden-div', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified')
)
def upload_data(contents, filename, date):
    if contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))

        df.insert(0, 'index', range(0, len(df)))
        return df.to_dict('records'), ''
    raise PreventUpdate

def clear_data():
    pass

@app.callback(
    Output('to-csv', 'data'),
    Input('download-data-button', 'n_clicks'),
    State('the-data', 'data')
)
def download_data(n_clicks, data):
    if n_clicks and data:
        df = pd.DataFrame.from_dict(data)
        return dcc.send_data_frame(df.to_csv, 'data.csv')