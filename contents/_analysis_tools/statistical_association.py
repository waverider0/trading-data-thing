import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler

from dash import  ALL
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from contents.app import *


@app.callback(
    Output('assoc-matrix-sliders-container', 'children'),
    Input('assoc-matrix-sliders', 'value'),
    State('the-data', 'data'),
)
def render_association_matrix_sliders(sliders, data):
    if sliders and data:
        df = pd.DataFrame.from_dict(data)
        children = []
        for slider in sliders:
            if slider == 'index':
                min = df.index.min()
                max = df.index.max()
            else:
                min = df[slider].min()
                max = df[slider].max()
            children.append(html.Div([
                html.B(slider, style={'margin-top': '5px', 'white-space': 'nowrap'}),
                html.Div(style={'margin-top': '5px'}),
                html.Div([
                    dcc.RangeSlider(
                        id={'type': 'assoc-matrix-slider', 'index': slider},
                        min=min,
                        max=max,
                        value=[min, max],
                        marks=None,
                        tooltip={'always_visible': False, 'placement': 'bottom'},
                    )
                ], style={'width': '100%', 'margin-top': '10px'})
            ], style={'display': 'flex'}))
        return children
    return []

@app.callback(
    Output('assoc-matrix-container', 'children'),
    Input('assoc-matrix-scale-data', 'on'),
    Input('assoc-matrix-metric', 'value'),
    Input('assoc-matrix-sliders', 'value'),
    Input({'type': 'assoc-matrix-slider', 'index': ALL}, 'value'),
    State('the-data', 'data')
)
def render_association_matrix(scale_data, metric, feature_filters, filter_ranges, data):
    if metric and data:
        df = pd.DataFrame.from_dict(data)
        if feature_filters and filter_ranges:
            for feature, range in zip(feature_filters, filter_ranges):
                if feature == 'index': df = df[(df.index >= range[0]) & (df.index <= range[1])]
                else: df = df[(df[feature] >= range[0]) & (df[feature] <= range[1])]
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

    return []

@app.callback(
    Output('joint-plot-sliders-container', 'children'),
    Input('joint-plot-sliders', 'value'),
    State('the-data', 'data'),
)
def render_joint_plot_sliders(sliders, data):
    if sliders and data:
        df = pd.DataFrame.from_dict(data)
        children = []
        for slider in sliders:
            if slider == 'index':
                min = df.index.min()
                max = df.index.max()
            else:
                min = df[slider].min()
                max = df[slider].max()
            children.append(html.Div([
                html.B(slider, style={'margin-top': '5px', 'white-space': 'nowrap'}),
                html.Div(style={'margin-top': '5px'}),
                html.Div([
                    dcc.RangeSlider(
                        id={'type': 'joint-plot-slider', 'index': slider},
                        min=min,
                        max=max,
                        value=[min, max],
                        marks=None,
                        tooltip={'always_visible': False, 'placement': 'bottom'},
                    )
                ], style={'width': '100%', 'margin-top': '10px'})
            ], style={'display': 'flex'}))
        return children
    return []

@app.callback(
    Output('joint-plot-container', 'children'),
    Input('joint-plot-feature-x', 'value'),
    Input('joint-plot-feature-y', 'value'),
    Input('joint-plot-sliders', 'value'),
    Input({'type': 'joint-plot-slider', 'index': ALL}, 'value'),
    State('the-data', 'data')
)
def render_joint_plot(feature_x, feature_y, feature_filters, filter_ranges, data):
    if feature_x and feature_y and data:
        df = pd.DataFrame.from_dict(data)
        if feature_filters and filter_ranges:
            for feature, range in zip(feature_filters, filter_ranges):
                if feature == 'index': df = df[(df.index >= range[0]) & (df.index <= range[1])]
                else: df = df[(df[feature] >= range[0]) & (df[feature] <= range[1])]
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
    return []

@app.callback(
    Output('heatmap-sliders-container', 'children'),
    Input('heatmap-sliders', 'value'),
    State('the-data', 'data')
)
def render_heatmap_sliders(sliders, data):
    if sliders and data:
        df = pd.DataFrame.from_dict(data)
        children = []
        for slider in sliders:
            if slider == 'index':
                min = df.index.min()
                max = df.index.max()
            else:
                min = df[slider].min()
                max = df[slider].max()
            children.append(html.Div([
                html.B(slider, style={'margin-top': '5px', 'white-space': 'nowrap'}),
                html.Div(style={'margin-top': '5px'}),
                html.Div([
                    dcc.RangeSlider(
                        id={'type': 'heatmap-slider', 'index': slider},
                        min=min,
                        max=max,
                        value=[min, max],
                        marks=None,
                        tooltip={'always_visible': False, 'placement': 'bottom'},
                    )
                ], style={'width': '100%', 'margin-top': '10px'})
            ], style={'display': 'flex'}))
        return children
    return []

@app.callback(
    Output('heatmap-container', 'children'),
    Input('heatmap-feature-x', 'value'),
    Input('heatmap-feature-y', 'value'),
    Input('heatmap-magnitude', 'value'),
    Input('heatmap-sliders', 'value'),
    Input({'type': 'heatmap-slider', 'index': ALL}, 'value'),
    Input('heatmap-colorscale', 'value'),
    Input('heatmap-reverse-colorscale', 'on'),
    State('the-data', 'data')
)
def render_heatmap(feature_x, feature_y, magnitude, feature_filters, filter_ranges, colorscale, reverse, data):
    if feature_x and feature_y and magnitude and colorscale and data:
        df = pd.DataFrame.from_dict(data)
        if feature_filters and filter_ranges:
            for feature, range in zip(feature_filters, filter_ranges):
                if feature == 'index': df = df[(df.index >= range[0]) & (df.index <= range[1])]
                else: df = df[(df[feature] >= range[0]) & (df[feature] <= range[1])]
        if magnitude == 'density':
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
                            colorscale=colorscale,
                            reversescale=reverse,
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
                            z=df[magnitude],
                            autobinx=True,
                            autobiny=True,
                            histfunc='avg',
                            zsmooth = 'best',
                            colorscale=colorscale,
                            reversescale=reverse,
                            colorbar=dict(title=magnitude)
                        )],
                        'layout': go.Layout(
                            title=f'{feature_x} vs {feature_y} vs {magnitude}',
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
    return []


# Helper Methods
def normalize_df(df):
    """ Normalize a dataframe with MinMaxScaler (Keep the column names) """
    cols = df.columns
    df = pd.DataFrame(MinMaxScaler().fit_transform(df))
    df.columns = cols
    return df