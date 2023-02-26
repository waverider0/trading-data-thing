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
def render_joint_plot(feature_x, feature_y, data):
    if feature_x and feature_y and data:
        df = pd.DataFrame.from_dict(data)
        return [
            dcc.Graph(
                id='joint-plot',
                figure=go.Figure(
                    data=[
                        go.Scatter(
                            x=df[feature_x],
                            y=df[feature_y],
                            mode='markers',
                            name='Data',
                            marker=dict(color='#37699b'),
                        ),
                        go.Scatter(
                            x=df[feature_x],
                            y=df[feature_x] * df[feature_x].corr(df[feature_y]) + df[feature_y].mean() - df[feature_x].mean() * df[feature_x].corr(df[feature_y]),
                            mode='lines',
                            line=dict(color='orange'),
                            name='OLS',
                        ),
                    ],
                    layout=go.Layout(
                        title=f'{feature_x} vs {feature_y}',
                        xaxis_title=feature_x,
                        yaxis_title=feature_y,
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

@app.callback(
    Output('heatmap-container', 'children'),
    Input('heatmap-feature-x', 'value'),
    Input('heatmap-feature-y', 'value'),
    Input('heatmap-heat', 'value'),
    State('the-data', 'data')
)
def render_heatmap(feature_x, feature_y, heat, data):
    if feature_x and feature_y and heat and data:
        df = pd.DataFrame.from_dict(data)
        if heat == 'density':
            return [
                dcc.Graph(
                    figure={
                        'data': [go.Histogram2d(
                            x=df[feature_x],
                            y=df[feature_y],
                            autobinx=True,
                            autobiny=True,
                            zsmooth = 'best',
                            histfunc='count',
                            colorscale='thermal',
                            reversescale=False,
                            colorbar=dict(title='Density')
                        )],
                        'layout': go.Layout(
                            title=f'Density ({feature_x} vs {feature_y})',
                            xaxis_title=feature_x,
                            yaxis_title=feature_y,
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
                    }
                )
            ]
        else:
            return [
                dcc.Graph(
                    figure={
                        'data': [go.Histogram2d(
                            x=df[feature_x],
                            y=df[feature_y],
                            z=df[heat],
                            autobinx=True,
                            autobiny=True,
                            histfunc='avg',
                            zsmooth = 'best',
                            colorscale='thermal',
                            reversescale=False,
                            colorbar=dict(title=heat)
                        )],
                        'layout': go.Layout(
                            title=f'{feature_x} vs {feature_y} vs {heat}',
                            xaxis_title=feature_x,
                            yaxis_title=feature_y,
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
                    }
                )
            ]
    raise PreventUpdate

# Helper Methods
def normalize_df(df):
    """ Normalize a dataframe with MinMaxScaler (Keep the column names) """
    cols = df.columns
    df = pd.DataFrame(MinMaxScaler().fit_transform(df))
    df.columns = cols
    return df