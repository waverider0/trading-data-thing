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
    Input('descriptive-stats-continuous-filters', 'value'),
    Input('descriptive-stats-categorical-filters', 'value'),
    Input({'type': 'descriptive-stats-continuous-filter', 'index': ALL}, 'value'),
    Input({'type': 'descriptive-stats-categorical-filter', 'index': ALL}, 'value'),
    State('the-data', 'data')
)
def calculate_descriptive_stat(
    feature,
    continous_filters,
    categorical_filters,
    filter_ranges,
    filter_categories,
    data
):
    if feature and data:
        df = pd.DataFrame.from_dict(data)
        if continous_filters and filter_ranges:
            for feature_, range in zip(continous_filters, filter_ranges):
                df = df[(df[feature_] >= range[0]) & (df[feature_] <= range[1])]
        if categorical_filters and filter_categories:
            if any(filter_categories):
                for feature_, categories in zip(categorical_filters, filter_categories):
                    if categories: df = df[df[feature_].isin(categories)]

        point_estimates = html.Div([
            html.Hr(),
            html.Div(style={'height': '17px'}),
            html.Div([
                html.B('Mean:'),
                html.Div(style={'width': '5px'}),
                html.Div(round(df[feature].mean(), 3))
            ], style={'display': 'flex'}),
            html.Div(style={'height': '5px'}),
            html.Div([
                html.B('Median:'),
                html.Div(style={'width': '5px'}),
                html.Div(round(df[feature].median(), 3))
            ], style={'display': 'flex'}),
            html.Div(style={'height': '5px'}),
            html.Div([
                html.B('Minimum:'),
                html.Div(style={'width': '5px'}),
                html.Div(round(df[feature].min(), 3))
            ], style={'display': 'flex'}),
            html.Div(style={'height': '5px'}),
            html.Div([
                html.B('Maximum:'),
                html.Div(style={'width': '5px'}),
                html.Div(round(df[feature].max(), 3))
            ], style={'display': 'flex'}),
            html.Div(style={'height': '5px'}),
            html.Div([
                html.B('Standard Deviation:'),
                html.Div(style={'width': '5px'}),
                html.Div(round(df[feature].std(), 3))
            ], style={'display': 'flex'}),
            html.Div(style={'height': '5px'}),
            html.Div([
                html.B('Skew:'),
                html.Div(style={'width': '5px'}),
                html.Div(round(df[feature].skew(), 3))
            ], style={'display': 'flex'}),
            html.Div(style={'height': '5px'}),
            html.Div([
                html.B('Kurtosis:'),
                html.Div(style={'width': '5px'}),
                html.Div(round(df[feature].kurt(), 3))
            ], style={'display': 'flex'}),
            html.Div(style={'height': '5px'}),
            html.Div([
                html.B('Count:'),
                html.Div(style={'width': '5px'}),
                html.Div(round(df[feature].count(), 3))
            ], style={'display': 'flex'}),
        ])

        histogram = dcc.Graph(
            figure={
                'data': [
                    go.Histogram(
                        x=df[feature],
                        name=feature,
                        histnorm='probability density',
                        marker=dict(color='#37699b'),
                    )
                ],
                'layout': go.Layout(
                    title='Empirical Distribution',
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
            }
        )

        return html.Div([
            point_estimates,
            histogram
        ])

    raise PreventUpdate

############
# Hit Rate #
############
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
    Input('hit-rate-continuous-filters', 'value'),
    Input('hit-rate-categorical-filters', 'value'),
    Input({'type': 'hit-rate-continuous-filter', 'index': ALL}, 'value'),
    Input({'type': 'hit-rate-categorical-filter', 'index': ALL}, 'value'),
    State('the-data', 'data')
)
def calculate_hit_rate(
    feature,
    continous_filters,
    categorical_filters,
    filter_ranges,
    filter_categories,
    data
):
    if feature and data:
        df = pd.DataFrame.from_dict(data)
        if continous_filters and filter_ranges:
            for feature_, range in zip(continous_filters, filter_ranges):
                df = df[(df[feature_] >= range[0]) & (df[feature_] <= range[1])]
        if categorical_filters and filter_categories:
            if any(filter_categories):
                for feature_, categories in zip(categorical_filters, filter_categories):
                    if categories: df = df[df[feature_].isin(categories)]

        children = [
            html.Hr(),
            html.Div(style={'height': '17px'}),
        ]

        # hit rates
        for unique in df[feature].unique():
            children.append(html.Div([
                html.Div([
                    html.B(f'{unique} Hit Rate:'),
                    html.Div(style={'width': '5px'}),
                    html.Div(round(df[df[feature] == unique].shape[0] / df.shape[0], 3))
                ], style={'display': 'flex'}),
                html.Div(style={'height': '5px'})
            ]))
        children.append(html.Div([
            html.B('Count:'),
            html.Div(style={'width': '5px'}),
            html.Div(df[feature].count())
        ], style={'display': 'flex'}))

        # histogram
        children.append(dcc.Graph(
            figure={
                'data': [
                    go.Histogram(
                        x=df[feature],
                        name=feature,
                        histnorm='probability density',
                        marker=dict(color='#37699b'),
                    )
                ],
                'layout': go.Layout(
                    title='Empirical Distribution',
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
            }
        ))

        return children

    raise PreventUpdate