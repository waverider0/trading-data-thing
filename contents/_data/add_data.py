import io
import base64
import numpy as np
import pandas as pd

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from contents.app import *

@app.callback(
    Output('the-data', 'data'),
    Output('datatable', 'data'),
    Output('datatable', 'columns'),
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
        df_dict = df.to_dict('records')

        return [
            df_dict,
            df_dict,
            [{'name': i, 'id': i} for i in df.columns],
        ]
    raise PreventUpdate

@app.callback(
    Output('datatable', 'data'),
    Output('datatable', 'columns'),
    Input('url', 'pathname'),
    State('the-data', 'data')
)
def render_table(path, data):
    if path == '/add-data':
        df = pd.DataFrame.from_dict(data)
        return [
            data,
            [{'name': i, 'id': i} for i in df.columns],
        ]
    raise PreventUpdate