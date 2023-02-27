import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler

from dash import  ALL
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from contents.app import *


@app.callback(
    Output('var-plot-sliders-container', 'children'),
    Input('var-plot-sliders', 'value'),
    State('the-data', 'data'),
)
def render_var_plot_sliders(sliders, data):
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
                        id={'type': 'var-plot-slider', 'index': slider},
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
    Output('var-plot-container', 'children'),
    Input('var-plot-type', 'value'),
    Input('var-plot-scale-data', 'on'),
    Input('var-plot-sliders', 'value'),
    Input({'type': 'var-plot-slider', 'index': ALL}, 'value'),
    State('the-data', 'data')
)
def render_var_plot(plot_type, scale_data, feature_filters, filter_ranges, data):
    if plot_type and data:
        df = pd.DataFrame.from_dict(data)
        if feature_filters and filter_ranges:
            for feature, range in zip(feature_filters, filter_ranges):
                if feature == 'index': df = df[(df.index >= range[0]) & (df.index <= range[1])]
                else: df = df[(df[feature] >= range[0]) & (df[feature] <= range[1])]
        if scale_data: df = normalize_df(df)

        if plot_type == 'box':
            return [
                dcc.Graph(
                    figure={
                        'data': [go.Box(y=df[col], name=col, boxpoints='outliers') for col in df.columns],
                        'layout': go.Layout(
                            title='Feature Box Plot',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#FFFFFF'),
                            xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=False),
                        ),
                    }
                ),
            ]
        elif plot_type == 'violin':
            return [
                dcc.Graph(
                    figure={
                        'data': [go.Violin(y=df[col], name=col, points='outliers', meanline_visible=True) for col in df.columns],
                        'layout': go.Layout(
                            title='Feature Violin Plot',
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
    Output('dist-plot-sliders-container', 'children'),
    Input('dist-plot-sliders', 'value'),
    State('the-data', 'data'),
)
def render_dist_plot_sliders(sliders, data):
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
                        id={'type': 'dist-plot-slider', 'index': slider},
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
    Output('dist-plot-container', 'children'),
    Input('dist-plot-scale-data', 'on'),
    Input('dist-plot-feature', 'value'),
    Input('dist-plot-distributions', 'value'),
    Input('dist-plot-sliders', 'value'),
    Input({'type': 'dist-plot-slider', 'index': ALL}, 'value'),
    State('the-data', 'data'),
)
def render_dist_plot(scale_data, feature, distributions, feature_filters, filter_ranges, data):
    if feature and distributions and data:
        df = pd.DataFrame.from_dict(data)
        if feature_filters and filter_ranges:
            for feature, range in zip(feature_filters, filter_ranges):
                if feature == 'index': df = df[(df.index >= range[0]) & (df.index <= range[1])]
                else: df = df[(df[feature] >= range[0]) & (df[feature] <= range[1])]
        if scale_data: df = normalize_df(df)

        graph = dcc.Graph(
            figure=go.Figure(
                layout=go.Layout(
                    title='Empirical vs Theoretical Distributions',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#FFFFFF'),
                    xaxis=dict(
                        title=feature,
                        titlefont=dict(color='#FFFFFF'),
                        showgrid=False,
                    ),
                    yaxis=dict(
                        title='Density',
                        titlefont=dict(color='#FFFFFF'),
                        showgrid=False,
                    ),
                )
            )
        )
        graph.figure.add_trace(
            go.Histogram(
                x=df[feature],
                name=feature,
                histnorm='probability density',
                marker=dict(color='#37699b'),
            )
        )
        for dist in distributions:
            if dist == 'normal':
                mu, std = norm.fit(df[feature])
                x = np.linspace(min(df[feature]), max(df[feature]), 100)
                p = norm.pdf(x, mu, std)
                graph.figure.add_trace(
                    go.Scatter(
                        x=x,
                        y=p,
                        mode='lines',
                        name='Normal',
                    )
                )
            elif dist == 'lognormal':
                pass
        
        return graph

    return []


# Helper Methods
def normalize_df(df):
    """ Normalize a dataframe with MinMaxScaler (Keep the column names) """
    cols = df.columns
    df = pd.DataFrame(MinMaxScaler().fit_transform(df))
    df.columns = cols
    return df