import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from contents.app import *


@app.callback(
    Output('line-plot-container', 'children'),
    Input('line-plot-scale-data', 'on'),
    Input('line-plot-features', 'value'),
    State('the-data', 'data')
)
def render_line_plot(scale_data, features, data):
    if features and data:
        df = pd.DataFrame.from_dict(data)
        if scale_data: df = normalize_df(df)
        return [
            dcc.Graph(
                figure={
                    'data': [go.Scatter(x=df.index, y=df[col], name=col) for col in features],
                    'layout': go.Layout(
                        title='Feature Line Plot',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#FFFFFF'),
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=False),
                    ),
                }
            ),
        ]
    raise PreventUpdate

@app.callback(
    Output('assoc-matrix-container', 'children'),
    Input('assoc-matrix-scale-data', 'on'),
    Input('assoc-matrix-metric', 'value'),
    State('the-data', 'data')
)
def render_association_matrix(scale_data, metric, data):
    if metric and data:
        df = pd.DataFrame.from_dict(data)
        if scale_data: df = normalize_df(df)

        matrix = pd.DataFrame()
        if metric == 'cov': matrix = df.cov()
        elif metric == 'pearson': matrix = df.corr(method='pearson')
        elif metric == 'spearman': matrix = df.corr(method='spearman')
        elif metric == 'kendall': matrix = df.corr(method='kendall')
        matrix = matrix.where(np.triu(np.ones(matrix.shape), k=1).astype(np.bool))
        
        return [
            dcc.Graph(
                figure={
                    'data': [go.Heatmap(
                        z=matrix.values,
                        x=matrix.columns,
                        y=matrix.columns,
                        colorscale='viridis',
                    )],
                    'layout': go.Layout(
                        title='Association Matrix',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#FFFFFF'),
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=False),
                    ),
                }
            ),
        ]

    raise PreventUpdate

@app.callback(
    Output('joint-plot-container', 'children'),
    Input('joint-plot-feature-x', 'value'),
    Input('joint-plot-feature-y', 'value'),
    State('the-data', 'data')
)
def render_joint_plot(feature_1, feature_2, data):
    if feature_1 and feature_2 and data:
        df = pd.DataFrame.from_dict(data)
        return [
            dcc.Graph(
                id='joint-plot',
                figure=go.Figure(
                    data=[
                        go.Scatter(
                            x=df[feature_1],
                            y=df[feature_2],
                            mode='markers',
                            name='Data',
                            marker=dict(color='#37699b'),
                        ),
                        go.Scatter(
                            x=df[feature_1],
                            y=df[feature_1] * df[feature_1].corr(df[feature_2]) + df[feature_2].mean() - df[feature_1].mean() * df[feature_1].corr(df[feature_2]),
                            mode='lines',
                            line=dict(color='orange'),
                            name='OLS',
                        ),
                    ],
                    layout=go.Layout(
                        title=f'{feature_1} vs {feature_2}',
                        xaxis_title=feature_1,
                        yaxis_title=feature_2,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#FFFFFF'),
                        xaxis=dict(
                            titlefont=dict(color='#FFFFFF'),
                            showgrid=False,
                        ),
                        yaxis=dict(
                            titlefont=dict(color='#FFFFFF'),
                            showgrid=False,
                        ),
                    ),
                )
            ),
        ]
    raise PreventUpdate


# Helper Methods
def normalize_df(df):
    """ Normalize a dataframe with MinMaxScaler (Keep the column names) """
    cols = df.columns
    df = pd.DataFrame(MinMaxScaler().fit_transform(df))
    df.columns = cols
    return df