import numpy as np
import pandas as pd

from dash import ALL
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash import dash_table

from contents.app import *

##########################
# Descriptive Statistics #
########################## 
@app.callback(
    Output('descriptive-stats-filters-container', 'children'),
    Input('descriptive-stats-continuous-filters', 'value'),
    Input('descriptive-stats-categorical-filters', 'value'),
    State('the-data', 'data')
)
def render_filters(continuous_filters, catagorical_filters, data):
    if data:
        df = pd.DataFrame.from_dict(data)
        children = []

        if continuous_filters:
            for filter in continuous_filters:
                min = df[filter].min()
                max = df[filter].max()
                children.append(html.Div([
                    html.B(filter, style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'margin-top': '5px'}),
                    html.Div([
                        dcc.RangeSlider(
                            id={'type': 'descriptive-stats-continuous-filter', 'index': filter},
                            min=min,
                            max=max,
                            value=[min, max],
                            marks=None,
                            tooltip={'always_visible': False, 'placement': 'bottom'},
                        )
                    ], style={'width': '100%', 'margin-top': '10px'}),
                ], style={'display': 'flex'}))
                
        if catagorical_filters:
            for filter in catagorical_filters:
                # checkboxes for each category
                children.append(html.Div([
                    html.Div([
                        html.B(filter, style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'margin-top': '5px'}),
                        html.Div([
                            dcc.Checklist(
                                id={'type': 'descriptive-stats-categorical-filter', 'index': filter},
                                options=[{'label': i, 'value': i} for i in df[filter].unique()],
                                labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                            )
                        ], style={'width': '100%', 'margin-top': '5px', 'margin-left': '20px'}),
                    ], style={'display': 'flex'}),
                    html.Div(style={'height': '17px'})
                ]))

        return children
    
    raise PreventUpdate

@app.callback(
    Output('descriptive-stats-output', 'children'),
    Input('descriptive-stats-feature', 'value'),
    Input('descriptive-stat', 'value'),
    Input('descriptive-stats-continuous-filters', 'value'),
    Input('descriptive-stats-categorical-filters', 'value'),
    Input({'type': 'descriptive-stats-continuous-filter', 'index': ALL}, 'value'),
    Input({'type': 'descriptive-stats-categorical-filter', 'index': ALL}, 'value'),
    State('the-data', 'data')
)
def calculate_descriptive_stat(
    feature,
    stat,
    continous_filters,
    categorical_filters,
    filter_ranges,
    filter_categories,
    data
):
    if feature and stat and data:
        df = pd.DataFrame.from_dict(data)
        if continous_filters and filter_ranges:
            for feature_, range in zip(continous_filters, filter_ranges):
                df = df[(df[feature_] >= range[0]) & (df[feature_] <= range[1])]
        if categorical_filters and filter_categories:
            if any(filter_categories):
                for feature_, categories in zip(categorical_filters, filter_categories):
                    df = df[df[feature_].isin(categories)]

        if stat == 'mean':
            return html.Div([
                html.B('Mean:'),
                html.Div(style={'width': '5px'}),
                html.Div(round(df[feature].mean(), 3))
            ], style={'display': 'flex'})
        elif stat == 'median':
            return html.Div([
                html.B('Median:'),
                html.Div(style={'width': '5px'}),
                html.Div(round(df[feature].median(), 3))
            ], style={'display': 'flex'})
        elif stat == 'std':
            return html.Div([
                html.B('Standard Deviation:'),
                html.Div(style={'width': '5px'}),
                html.Div(round(df[feature].std(), 3))
            ], style={'display': 'flex'})
        elif stat == 'min':
            return html.Div([
                html.B('Minimum:'),
                html.Div(style={'width': '5px'}),
                html.Div(round(df[feature].min(), 3))
            ], style={'display': 'flex'})
        elif stat == 'max':
            return html.Div([
                html.B('Maximum:'),
                html.Div(style={'width': '5px'}),
                html.Div(round(df[feature].max(), 3))
            ], style={'display': 'flex'})
        elif stat == 'skew':
            return html.Div([
                html.B('Skew:'),
                html.Div(style={'width': '5px'}),
                html.Div(round(df[feature].skew(), 3))
            ], style={'display': 'flex'})
        elif stat == 'kurt':
            return html.Div([
                html.B('Kurtosis:'),
                html.Div(style={'width': '5px'}),
                html.Div(round(df[feature].kurt(), 3))
            ], style={'display': 'flex'})
    
    raise PreventUpdate

############
# Hit Rate #
############
@app.callback(
    Output('hit-rate-class', 'options'),
    Input('hit-rate-feature', 'value'),
    State('the-data', 'data')
)
def update_class(feature, data):
    if feature and data:
        df = pd.DataFrame.from_dict(data)
        return [{'label': i, 'value': i} for i in df[feature].unique()]
    raise PreventUpdate

@app.callback(
    Output('hit-rate-filters-container', 'children'),
    Input('hit-rate-continuous-filters', 'value'),
    Input('hit-rate-categorical-filters', 'value'),
    State('the-data', 'data')
)
def render_filters(continuous_filters, catagorical_filters, data):
    if data:
        df = pd.DataFrame.from_dict(data)
        children = []

        if continuous_filters:
            for filter in continuous_filters:
                min = df[filter].min()
                max = df[filter].max()
                children.append(html.Div([
                    html.B(filter, style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'margin-top': '5px'}),
                    html.Div([
                        dcc.RangeSlider(
                            id={'type': 'hit-rate-continuous-filter', 'index': filter},
                            min=min,
                            max=max,
                            value=[min, max],
                            marks=None,
                            tooltip={'always_visible': False, 'placement': 'bottom'},
                        )
                    ], style={'width': '100%', 'margin-top': '10px'}),
                ], style={'display': 'flex'}))
                
        if catagorical_filters:
            for filter in catagorical_filters:
                # checkboxes for each category
                children.append(html.Div([
                    html.Div([
                        html.B(filter, style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'margin-top': '5px'}),
                        html.Div([
                            dcc.Checklist(
                                id={'type': 'hit-rate-categorical-filter', 'index': filter},
                                options=[{'label': i, 'value': i} for i in df[filter].unique()],
                                labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                            )
                        ], style={'width': '100%', 'margin-top': '5px', 'margin-left': '20px'}),
                    ], style={'display': 'flex'}),
                    html.Div(style={'height': '17px'})
                ]))

        return children
    
    raise PreventUpdate

@app.callback(
    Output('hit-rate-output', 'children'),
    Input('hit-rate-feature', 'value'),
    Input('hit-rate-class', 'value'),
    Input('hit-rate-continuous-filters', 'value'),
    Input('hit-rate-categorical-filters', 'value'),
    Input({'type': 'hit-rate-continuous-filter', 'index': ALL}, 'value'),
    Input({'type': 'hit-rate-categorical-filter', 'index': ALL}, 'value'),
    State('the-data', 'data')
)
def calculate_hit_rate(
    feature,
    _class,
    continous_filters,
    categorical_filters,
    filter_ranges,
    filter_categories,
    data
):
    if feature and _class and data:
        df = pd.DataFrame.from_dict(data)
        if continous_filters and filter_ranges:
            for feature_, range in zip(continous_filters, filter_ranges):
                df = df[(df[feature_] >= range[0]) & (df[feature_] <= range[1])]
        if categorical_filters and filter_categories:
            if any(filter_categories):
                for feature_, categories in zip(categorical_filters, filter_categories):
                    df = df[df[feature_].isin(categories)]

        return html.Div([
            html.B('Hit Rate:'),
            html.Div(style={'width': '5px'}),
            html.Div(round(df[df[feature] == _class].shape[0] / df.shape[0], 3))
        ], style={'display': 'flex'})

    raise PreventUpdate