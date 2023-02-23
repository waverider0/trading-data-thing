import pandas as pd

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from contents.app import *

@app.callback(
    # Table
    Output('preprocessing-table', 'data'),
    Output('preprocessing-table', 'columns'),
    # Plots
    Output('missing-values-plot', 'figure'),
    # Inputs
    Input('preprocessing-refresh', 'n_clicks'),
    State('the-data', 'data')
)
def render_table(n_clicks, data):
    if n_clicks and data:
        df = pd.DataFrame.from_dict(data)
        missing_values = df.isnull().sum()
        return [
            # Table
            data,
            [{'name': i, 'id': i} for i in df.columns],
            # Plots
            go.Figure(
                data=[
                    go.Bar(
                        x=missing_values.index,
                        y=missing_values.values,
                        name='Number of Nulls'
                    ),
                ],
                layout=go.Layout(
                    title='Missing Values',
                    xaxis=dict(
                        titlefont=dict(color='#FFFFFF'),
                        showgrid=False,
                    ),
                    yaxis=dict(
                        titlefont=dict(color='#FFFFFF'),
                        showgrid=False,
                    ),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#FFFFFF'),
                ),
            ),
        ]
    raise PreventUpdate

@app.callback(
    Output('missing-values-plot', 'figure'),
    Output('the-data', 'data'),
    Output('preprocessing-table', 'data'),
    Output('preprocessing-table', 'columns'),
    Input('remove-nulls-button', 'n_clicks'),
    State('the-data', 'data')
)
def remove_nulls(n_clicks, data):
    if n_clicks:
        df = pd.DataFrame.from_dict(data).dropna()
        missing_values = df.isnull().sum()
        return [
            go.Figure(
                data=[
                    go.Bar(
                        x=missing_values.index,
                        y=missing_values.values,
                        name='Number of Nulls'
                    ),
                ],
                layout=go.Layout(
                    title='Missing Values',
                    xaxis=dict(
                        titlefont=dict(color='#FFFFFF'),
                        showgrid=False,
                    ),
                    yaxis=dict(
                        titlefont=dict(color='#FFFFFF'),
                        showgrid=False,
                    ),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#FFFFFF'),
                ),
            ),
            df.to_dict('records'),
            df.to_dict('records'),
            [{'name': col, 'id': col} for col in df.columns],
            True
        ]
    raise PreventUpdate