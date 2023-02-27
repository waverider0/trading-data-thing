import math
import os
import pandas as pd

from dash import html, dcc
from dash_extensions.enrich import DashProxy, MultiplexerTransform
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from dash import dash_table
import dash_daq as daq
import dash_bootstrap_components as dbc
import dash_loading_spinners as dls
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import plotly.graph_objs as go

from contents.storage.utils import get_saved_models


##################
# Module Layouts #
##################
add_data = [
    html.H2('Add Data', id='header'),
    dmc.Container([
        # datatable
        dash_table.DataTable(
            id='datatable',
            columns=[{"name": i, "id": i} for i in ['']],
            data=[],
            page_size=15,
            style_table={'overflowX': 'scroll'},
        ),
        # upload button
        dcc.Upload(
            id='upload-data',
            children=html.Div('Drag and Drop or Select Files'),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin-top': '10px'
            },
            multiple=False
        ),
        # Clear Data and Download Data buttons
        html.Div(style={'height': '10px'}),
        html.Div([
            dbc.Button(
                '🗑️ Clear Data',
                id='clear-data-button',
                color='primary',
                style={'width': '100%'}
            ),
            html.Div(style={'width': '20px'}),
            dbc.Button(
                '📁 Download Data',
                id='download-data-button',
                color='primary',
                style={'width': '100%'}
            )
        ], style={'display': 'flex'}),
    ],
    id='add-data-container',
    fluid=True,
    style={
        'border-style': 'solid',
        'border-color': 'grey',
        'border-width': '1px',
        'padding': '12px',
    }
    ),
    dbc.Popover(
        [],
        target='add-data-container',
        trigger='hover',
        hide_arrow=True,
        id='add-data-container-popover',
    ),
    html.Div(style={'height': '10px'}),
]
add_features = [
    html.H2('Add Features'),
    dmc.Container([
        dbc.Accordion([
            dbc.AccordionItem(
                title="Drop Features",
                children=[
                    html.Div([
                        html.B('Features', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Dropdown(
                            id='drop-features-dropdown',
                            options=[{'label': i, 'value': i} for i in []],
                            multi=True,
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '30px'}),
                        dbc.Button(
                            'Drop Features',
                            id='drop-features-button',
                            color='primary',
                            style={'width': '100%'}
                        )
                    ], style={'display': 'flex'})
                ]
            ),
            dbc.AccordionItem(
                title="Rename Features",
                children=[
                    html.Div([
                        html.B('Feature', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Dropdown(
                            id='rename-features-dropdown',
                            options=[],
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '20px'}),
                        html.B('New Name', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Input(
                            id='feature-new-name',
                            type='text',
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '20px'}),
                        dbc.Button(
                            'Rename',
                            id='rename-feature-button',
                            color='primary',
                            style={'width': '100%'}
                        )
                    ], style={'display': 'flex'}),
                ]
            ),
            dbc.AccordionItem(
                title="Type Casting",
                children=html.Div([
                    html.B('Feature', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Dropdown(
                        id='type-casting-feature-dropdown',
                        options=[{'label': i, 'value': i} for i in []],
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '30px'}),
                    html.B('Type', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Dropdown(
                        id='type-casting-type-dropdown',
                        options=[
                            {'label': 'Integer', 'value': 'int'},
                            {'label': 'Float', 'value': 'float'},
                            {'label': 'String', 'value': 'str'},
                            {'label': 'Boolean', 'value': 'bool'},
                            {'label': 'Datetime', 'value': 'datetime'},
                        ],
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '30px'}),
                    dbc.Button(
                        'Apply',
                        id='type-casting-apply-button',
                        color='primary',
                        style={'width': '100%'}
                    )
                ], style={'display': 'flex'})
            ),
            dbc.AccordionItem(
                title="Order By",
                children=[
                    html.Div([
                        html.B('Feature', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Dropdown(
                            id='order-by-feature-dropdown',
                            options=[],
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '30px'}),
                        html.B('Order', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Dropdown(
                            id='order-by-order-dropdown',
                            options=[
                                {'label': 'Ascending', 'value': 'ascending'},
                                {'label': 'Descending', 'value': 'descending'},
                            ],
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '30px'}),
                        dbc.Button(
                            'Apply',
                            id='order-by-apply-button',
                            color='primary',
                            style={'width': '100%'}
                        )
                    ], style={'display': 'flex'}),
                ]
            ),
            dbc.AccordionItem(
                title="Transformations",
                children=[
                    html.Div([
                        html.B('Base feature(s)', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Dropdown(
                            id='transformations-base-features',
                            options=[{'label': i, 'value': i} for i in []],
                            multi=True,
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '30px'}),
                        html.B('Transformation(s)', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Dropdown(
                            id='transformations-dropdown',
                            options=[
                                {'label': 'Lag', 'value': 'lag'},
                                {'label': 'Nth Power', 'value': 'nth_power'},
                                {'label': 'Nth Root', 'value': 'nth_root'},
                                {'label': 'Nth Derivative', 'value': 'nth_deriv'},
                                {'label': 'Log Base N', 'value': 'log_n'},
                                {'label': 'Absolute Value', 'value': 'abs'},
                                {'label': 'Percent Change', 'value': 'pct_change'},
                                {'label': 'Log Return', 'value': 'log_ret'},
                                {'label': 'Rolling Min', 'value': 'min'},
                                {'label': 'Rolling Max', 'value': 'max'},
                                {'label': 'Min Position', 'value': 'min_pos'},
                                {'label': 'Max Position', 'value': 'max_pos'},
                                {'label': 'Rolling Sum', 'value': 'sum'},
                                {'label': 'SMA', 'value': 'sma'},
                                {'label': 'EMA', 'value': 'ema'},
                                {'label': 'Rolling Median', 'value': 'median'},
                                {'label': 'Rolling Mode', 'value': 'mode'},
                                {'label': 'Rolling Stdev', 'value': 'std'},
                                {'label': 'Rolling Variance', 'value': 'var'},
                                {'label': 'Rolling Skew', 'value': 'skew'},
                                {'label': 'Rolling Kurtosis', 'value': 'kurt'},
                                {'label': 'Rolling Z-Score', 'value': 'z_score'},
                                {'label': 'Rolling Correlation', 'value': 'corr'},
                                {'label': 'Rolling Autocorrelation', 'value': 'autocorr'},
                                #{'label': 'CUSUM Filter', 'value': 'cusum'},
                                {'label': 'Kalman Filter', 'value': 'kalman'},
                            ],
                            multi=True,
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '30px'}),
                        html.B('Window/Input', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Input(
                            id='transformations-input',
                            type='number',
                            value=1,
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '30px'}),
                        dbc.Button(
                            'Add Feature',
                            id='add-features-button',
                            color='primary',
                            style={'width': '100%'}
                        )
                    ], style={'display': 'flex'})
                ]
            ),
            dbc.AccordionItem(
                title="OHLCV Features",
                children=[
                    html.Div([
                        html.B('Open', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Dropdown(
                            id='open-feature-dropdown',
                            options=[],
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '30px'}),
                        html.B('High', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Dropdown(
                            id='high-feature-dropdown',
                            options=[],
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '30px'}),
                        html.B('Low', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Dropdown(
                            id='low-feature-dropdown',
                            options=[],
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '30px'}),
                        html.B('Close', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Dropdown(
                            id='close-feature-dropdown',
                            options=[],
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '30px'}),
                        html.B('Volume', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Dropdown(
                            id='volume-feature-dropdown',
                            options=[],
                            style={'width': '100%'}
                        ),
                    ], style={'display': 'flex'}),
                    html.Div(style={'height': '10px'}),
                    html.Div([
                        html.B('Features', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Dropdown(
                            id='ohlcv-features-dropdown',
                            options=[
                                # Oscillators
                                {'label': 'RSI', 'value': 'rsi'},
                                # Volatility Estimators
                                {'label': 'ATR', 'value': 'atr'},
                                {'label': 'Close-To-Close Volatility', 'value': 'c2c_vol'},
                                {'label': 'Parkinson Volatility', 'value': 'parkinson_vol'},
                                {'label': 'Garman Klass Volatility', 'value': 'garman_klass_vol'},
                                {'label': 'Rodgers Satchell Volatility', 'value': 'rodgers_satchell_vol'},
                                {'label': 'Yang Zhang Volatility', 'value': 'yang_zhang_vol'},
                                # {'label': 'First Exit Time Volatility', 'value': 'fet_vol'},
                                # Filters
                                # {'label': 'CUSUM Filter', 'value': 'cusum'},
                                # {'label': 'Z-Score Filter', 'value': 'zscore'},
                                # Fractional Differentiation
                                # {'label': 'Fractional Differentiation', 'value': 'frac_diff'},
                            ],
                            multi=True,
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '20px'}),
                        html.B('Window', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Input(
                            id='ohlcv-features-window',
                            type='number',
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '20px'}),
                        dbc.Button(
                            'Add Features',
                            id='add-ohlcv-features-button',
                            color='primary',
                            style={'width': '100%'}
                        )
                    ], style={'display': 'flex'})
                ]
            ),
            dbc.AccordionItem(
                title="Filters",
            ),
            dbc.AccordionItem(
                title="Metalabeling",
            ),
        ], start_collapsed=True, always_open=False),
    ], 
    id='add-features-container',
    fluid=True,
    style={
        'border-style': 'solid',
        'border-color': 'grey',
        'border-width': '1px',
        'padding': '12px',
    }
    ),
    dbc.Popover(
        [],
        target='add-features-container',
        trigger='hover',
        hide_arrow=True,
        id='add-features-container-popover',
    ),
    html.Div(style={'height': '10px'}),
]
preprocessing = [
    html.H2('Preprocessing'),
    dmc.Container([
        dbc.Accordion([
            dbc.AccordionItem(
                title="Missing Values",
                children=[
                    dcc.Graph(
                        id='missing-values-plot',
                        figure=go.Figure(
                            data=[],
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
                        )
                    ),
                    html.Br(),
                    dbc.Button(
                        'Remove Nulls',
                        id='remove-nulls-button',
                        color='primary',
                        style={'width': '100%'}
                    )
                ]
            ),
            dbc.AccordionItem(
                title="Outliers",
            ),
            dbc.AccordionItem(
                title="Scaling",
            ),
        ], start_collapsed=True, always_open=False)
    ],
    id='preprocessing-container',
    fluid=True,
    style={
        'border-style': 'solid',
        'border-color': 'grey',
        'border-width': '1px',
        'padding': '12px',
    }
    ),
    dbc.Popover(
        [],
        target='preprocessing-container',
        trigger='hover',
        hide_arrow=True,
        id='preprocessing-container-popover',
    ),
]

distributions = [
    html.H2('Distributions'),
    dmc.Container([
        dbc.Accordion([
            dbc.AccordionItem(
                title="Variance Plots",
                children=[
                    html.Div([
                        html.B('Scale Data', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '5px'}),
                        daq.BooleanSwitch(
                            id='var-plot-scale-data',
                            on=False,
                            style={'margin-top': '5px'}
                        ),
                        html.Div(style={'width': '10px'}),
                        html.B('Select Plot Type', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '5px'}),
                        dcc.Dropdown(
                            id='var-plot-type',
                            options=[
                                {'label': 'Box', 'value': 'box'},
                                {'label': 'Violin', 'value': 'violin'},
                            ],
                            value='scatter',
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '10px'}),
                        html.B('Sliders', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '5px'}),
                        dcc.Dropdown(
                            id='var-plot-sliders',
                            options=[],
                            multi=True,
                            style={'width': '100%'}
                        )
                    ], style={'display': 'flex'}),
                    html.Div(style={'height': '20px'}),
                    html.Div(id='var-plot-sliders-container'),
                    html.Div(style={'height': '20px'}),
                    html.Div(id='var-plot-container'),
                ],
            ),
            dbc.AccordionItem(
                title="Distribution Plots",
                children=[
                    html.Div([
                        html.B('Scale Data', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '5px'}),
                        daq.BooleanSwitch(
                            id='dist-plot-scale-data',
                            on=False,
                            style={'margin-top': '5px'}
                        ),
                        html.Div(style={'width': '10px'}),
                        html.B('Select Feature', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '5px'}),
                        dcc.Dropdown(
                            id='dist-plot-feature',
                            options=[],
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '10px'}),
                        html.B('Select Distributions', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '5px'}),
                        dcc.Dropdown(
                            id='dist-plot-distributions',
                            options=[
                                {'label': 'Normal', 'value': 'normal'},
                                {'label': 'Log Normal', 'value': 'lognormal'},
                                {'label': 'Exponential', 'value': 'exponential'},
                                {'label': 'Poisson', 'value': 'poisson'},
                                {'label': 'Gamma', 'value': 'gamma'},
                                {'label': 'Gumbel', 'value': 'gumbel'},
                            ],
                            multi=True,
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '10px'}),
                        html.B('Sliders', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '5px'}),
                        dcc.Dropdown(
                            id='dist-plot-sliders',
                            options=[],
                            multi=True,
                            style={'width': '100%'}
                        )
                    ], style={'display': 'flex'}),
                    html.Div(style={'height': '20px'}),
                    html.Div(id='dist-plot-sliders-container'),
                    html.Div(style={'height': '20px'}),
                    html.Div(id='dist-plot-container'),
                ],
            ),
            dbc.AccordionItem(
                title="QQ Plots",
            ),
            dbc.AccordionItem(
                title="Normality Tests",
            ),
        ], start_collapsed=True, always_open=False),
    ],
    id='distributions-container',
    fluid=True,
    style={
        'border-style': 'solid',
        'border-color': 'grey',
        'border-width': '1px',
        'padding': '12px',
    }
    ),
    dbc.Popover(
        [],
        target='distributions-container',
        trigger='hover',
        hide_arrow=True,
        id='distributions-container-popover',
    ),
]
statistical_association = [
    html.H2('Statistical Association'),
    dmc.Container([
        dbc.Accordion([
            dbc.AccordionItem(
                title="Association Matrices",
                children=[
                    html.Div([
                        html.B('Scale Data', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '5px'}),
                        daq.BooleanSwitch(
                            id='assoc-matrix-scale-data',
                            on=False,
                            style={'margin-top': '5px'}
                        ),
                        html.Div(style={'width': '10px'}),
                        html.B('Select Association Metric', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '5px'}),
                        dcc.Dropdown(
                            id='assoc-matrix-metric',
                            options=[
                                {'label': 'Covariance', 'value': 'cov'},
                                {'label': 'Pearson Correlation', 'value': 'pearson'},
                                {'label': 'Spearman Rho', 'value': 'spearman'},
                                {'label': 'Kendall Tau', 'value': 'kendall'},
                                {'label': 'Mutual Information', 'value': 'mutual_info'},
                                {'label': 'KSG Mutual Information', 'value': 'ksg'},
                                {'label': 'Maximal Information Coefficient', 'value': 'mic'},
                                {'label': 'Copula Distance', 'value': 'copula'},
                            ],
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '10px'}),
                        html.B('Sliders', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '5px'}),
                        dcc.Dropdown(
                            id='assoc-matrix-sliders',
                            options=[],
                            multi=True,
                            style={'width': '100%'}
                        )
                    ], style={'display': 'flex'}),
                    html.Div(style={'height': '20px'}),
                    html.Div(id='assoc-matrix-sliders-container'),
                    html.Div(style={'height': '20px'}),
                    html.Div(id='assoc-matrix-container'),
                ]
            ),
            dbc.AccordionItem(
                title="Joint Plots",
                children=[
                    html.Div([
                        html.B('X Feature', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '5px'}),
                        dcc.Dropdown(
                            id='joint-plot-feature-x',
                            options=[],
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '20px'}),
                        html.B('Y Feature', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '5px'}),
                        dcc.Dropdown(
                            id='joint-plot-feature-y',
                            options=[],
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '20px'}),
                        html.B('Sliders', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '5px'}),
                        dcc.Dropdown(
                            id='joint-plot-sliders',
                            options=[],
                            multi=True,
                            style={'width': '100%'}
                        )
                    ], style={'display': 'flex'}),
                    html.Div(style={'height': '20px'}),
                    html.Div(id='joint-plot-sliders-container'),
                    html.Div(style={'height': '20px'}),
                    html.Div(id='joint-plot-container'),
                ]
            ),
            dbc.AccordionItem(
                title="Heatmaps",
                children=[
                    html.Div([
                        html.B('Feature X', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '5px'}),
                        dcc.Dropdown(
                            id='heatmap-feature-x',
                            options=[],
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '20px'}),
                        html.B('Feature Y', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '5px'}),
                        dcc.Dropdown(
                            id='heatmap-feature-y',
                            options=[],
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '20px'}),
                        html.B('Magnitude', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '5px'}),
                        dcc.Dropdown(
                            id='heatmap-magnitude',
                            options=[],
                            style={'width': '100%'}
                        ),
                    ], style={'display': 'flex'}),
                    html.Div(style={'height': '10px'}),
                    html.Div([
                        html.B('Sliders', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '5px'}),
                        dcc.Dropdown(
                            id='heatmap-sliders',
                            options=[],
                            multi=True,
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '10px'}),
                        html.B('Colorscale', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '5px'}),
                        dcc.Dropdown(
                            id='heatmap-colorscale',
                            options=[
                                {'label': 'Thermal', 'value': 'thermal'},
                                {'label': 'Viridis', 'value': 'viridis'},
                                {'label': 'Portland', 'value': 'portland'},
                                {'label': 'Spectral', 'value': 'spectral'},
                                {'label': 'Red/Green', 'value': 'piyg'},
                                {'label': 'Red/Blue', 'value': 'RdBu'},
                            ],
                            value='thermal',
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '20px'}),
                        html.B('Reverse Colorscale', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '5px'}),
                        daq.BooleanSwitch(
                            id='heatmap-reverse-colorscale',
                            on=False,
                            style={'margin-top': '5px'}
                        ),
                    ], style={'display': 'flex'}),
                    html.Div(style={'height': '20px'}),
                    html.Div(id='heatmap-sliders-container'),
                    html.Div(style={'height': '20px'}),
                    html.Div(id='heatmap-container'),
                ]
            ),
            dbc.AccordionItem(
                title="3D Plots",
                children=[],
            ),
        ], start_collapsed=True, always_open=False),
    ],
    id='statistical-association-container',
    fluid=True,
    style={
        'border-style': 'solid',
        'border-color': 'grey',
        'border-width': '1px',
        'padding': '12px',
    }
    ),
    dbc.Popover(
        [],
        target='statistical-association-container',
        trigger='hover',
        hide_arrow=True,
        id='statistical-association-container-popover',
    ),

]
temporal_sequence = [
    html.H2('Temporal Sequence'),
    dmc.Container([
        dbc.Accordion([
            dbc.AccordionItem(
                title='Dynamic Time Warping',
            ),
            dbc.AccordionItem(
                title="Line Plot",
                children=[
                    html.Div([
                        html.B('Scale Data', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '5px'}),
                        daq.BooleanSwitch(
                            id='line-plot-scale-data',
                            on=False,
                            style={'margin-top': '5px'}
                        ),
                        html.Div(style={'width': '10px'}),
                        html.B('Select Features', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '5px'}),
                        dcc.Dropdown(
                            id='line-plot-features',
                            options=[],
                            multi=True,
                            style={'width': '100%'}
                        ),
                    ], style={'display': 'flex'}),
                    html.Div(style={'height': '10px'}),
                    html.Div(id='line-plot-container'),
                ]
            ),
            dbc.AccordionItem(
                title='Feature Drift',
                children=[
                    html.Div([
                        html.B('Test', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '5px'}),
                        dcc.Dropdown(
                            id='drift-plot-test',
                            options=[
                                {'label': 'None', 'value': 'none'},
                                {'label': 'Kolmogorov-Smirnov', 'value': 'ks'},
                                {'label': 'Cramer-von Mises', 'value': 'cvm'},
                            ],
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '20px'}),
                        html.B('Feature', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '5px'}),
                        dcc.Dropdown(
                            id='drift-plot-feature',
                            options=[],
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '20px'}),
                        html.B('N Splits', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '5px'}),
                        html.Div([
                            dcc.Slider(
                                id='drift-plot-n-splits',
                                min=2,
                                max=10,
                                step=1,
                                value=5,
                                marks=None,
                                tooltip={'always_visible': False, 'placement': 'bottom'},
                            ),
                        ], style={'width': '100%', 'margin-top': '10px'}),
                    ], style={'display': 'flex'}),
                    html.Div(style={'height': '10px'}),
                    html.Div(id='drift-plot-container'),
                ]
            ),
        ], start_collapsed=True, always_open=False),
    ],
    id='temporal-sequence-container',
    fluid=True,
    style={
        'border-style': 'solid',
        'border-color': 'grey',
        'border-width': '1px',
        'padding': '12px',
    }
    ),
    dbc.Popover(
        [],
        target='temporal-sequence-container',
        trigger='hover',
        hide_arrow=True,
        id='temporal-sequence-container-popover',
    ),
]

feature_importance = [
    html.H2('Feature Importance'),
    dmc.Container([
        dbc.Accordion([
            dbc.AccordionItem(
                title='Recursive Feature Elimination',
                children=[
                    html.Div([
                        html.B('Model Type', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Dropdown(
                            id='rfe-model-type',
                            options=[
                                {'label': 'Classification', 'value': 'clf'},
                                {'label': 'Regression', 'value': 'reg'},
                            ],
                            style={'width': '100%'}
                        ),
                    ], style={'display': 'flex'}),
                    html.Div(style={'height': '10px'}),
                    html.Div(id='rfe-inputs-container'),
                    html.Div(style={'height': '10px'}),
                    html.Div(id='rfe-plot-container'),
                ]
            ),
            dbc.AccordionItem(
                title='Sequential Feature Selection',
                children=[
                    html.Div([
                        html.B('Model Type', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Dropdown(
                            id='sfs-model-type',
                            options=[
                                {'label': 'Classification', 'value': 'clf'},
                                {'label': 'Regression', 'value': 'reg'},
                            ],
                            style={'width': '100%'}
                        ),
                    ], style={'display': 'flex'}),
                    html.Div(style={'height': '10px'}),
                    html.Div(id='sfs-inputs-container'),
                    html.Div(style={'height': '10px'}),
                    html.Div(id='sfs-plot-container'),
                ]
            ),
            dbc.AccordionItem(
                title='Boruta SHAP',
                children=[
                    html.Div([
                        html.B('Model Type', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Dropdown(
                            id='boruta-shap-model-type',
                            options=[
                                {'label': 'Classification', 'value': 'clf'},
                                {'label': 'Regression', 'value': 'reg'},
                            ],
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '20px'}),
                        html.B('Target', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Dropdown(
                            id='boruta-shap-target',
                            options=[],
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '20px'}),
                        html.B('Number of trials', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        html.Div([dcc.Slider(
                            id='boruta-shap-n-trials',
                            min=1,
                            max=100,
                            step=1,
                            value=10,
                            marks=None,
                            tooltip={'always_visible': False, 'placement': 'bottom'},
                        )], style={'width': '100%', 'margin-top': '10px'}),
                    ], style={'display': 'flex'}),
                    html.Div(style={'height': '10px'}),
                    dbc.Button(
                        'Run Boruta SHAP',
                        id='boruta-shap-run',
                        color='primary',
                        style={'width': '100%'},
                    ),
                    html.Div(style={'height': '10px'}),
                    html.Div(id='boruta-shap-plot-container'),
                ]
            ),
        ], start_collapsed=True, always_open=False),
    ],
    id='feature-importance-container',
    fluid=True,
    style={
        'border-style': 'solid',
        'border-color': 'grey',
        'border-width': '1px',
        'padding': '12px',
    }
    ),
    dbc.Popover(
        [],
        target='feature-importance-container',
        trigger='hover',
        hide_arrow=True,
        id='feature-importance-container-popover',
    ),
]
dimensionality_reduction = [
    html.H2('Dimensionality Reduction'),
    dmc.Container([
        dbc.Accordion([
            dbc.AccordionItem(
                title='Matrix Decomposition',
                children=[
                    html.Div([
                        html.B('Method', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Dropdown(
                            id='matrix-decomp-method',
                            options=[
                                {'label': 'PCA', 'value': 'pca'},
                                {'label': 'Kernel PCA', 'value': 'kpca'},
                                {'label': 'Truncated SVD', 'value': 'svd'},
                            ],
                            style={'width': '100%'}
                        ),
                    ], style={'display': 'flex'}),
                    html.Div(style={'height': '10px'}),
                    html.Div(id='matrix-decomp-inputs-container'),
                    html.Div(style={'height': '10px'}),
                    html.Div(id='matrix-decomp-plot-container'),
                ]
            ),
            dbc.AccordionItem(
                title='Manifold Learning',
            ),
            dbc.AccordionItem(
                title='Discriminant Analysis',
            ),
        ], start_collapsed=True, always_open=False),
    ],
    id='dimensionality-reduction-container',
    fluid=True,
    style={
        'border-style': 'solid',
        'border-color': 'grey',
        'border-width': '1px',
        'padding': '12px',
    }
    ),
    dbc.Popover(
        [],
        target='dimensionality-reduction-container',
        trigger='hover',
        hide_arrow=True,
        id='dimensionality-reduction-container-popover',
    ),
]
hyperparameter_tuning = [
    html.H2('Hyperparameter Tuning'),
    dmc.Container([
        dbc.Accordion([
            dbc.AccordionItem(
                title='Random Search',
                children=[
                    html.Div([
                        html.B('Model Type', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Dropdown(
                            id='rand-search-model-type',
                            options=[
                                {'label': 'Classification', 'value': 'clf'},
                                {'label': 'Regression', 'value': 'reg'},
                            ],
                            style={'width': '100%'}
                        ),
                    ], style={'display': 'flex'}),
                    html.Div(style={'height': '10px'}),
                    html.Div(id='rand-search-inputs-container'),
                    html.Div(style={'height': '10px'}),
                    html.Div(id='rand-search-plot-container'),
                ],
            ),
            dbc.AccordionItem(
                title='Bayesian Search',
                children=[
                    html.Div([
                        html.B('Model Type', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Dropdown(
                            id='bayes-search-model-type',
                            options=[
                                {'label': 'Classification', 'value': 'clf'},
                                {'label': 'Regression', 'value': 'reg'},
                            ],
                            style={'width': '100%'}
                        ),
                    ], style={'display': 'flex'}),
                    html.Div(style={'height': '10px'}),
                    html.Div(id='bayes-search-inputs-container'),
                    html.Div(style={'height': '10px'}),
                    html.Div(id='bayes-search-plot-container'),
                ],
            ),
            dbc.AccordionItem(
                title='Evolutionary Search',
                children=[
                    html.Div([
                        html.B('Model Type', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Dropdown(
                            id='evo-search-model-type',
                            options=[
                                {'label': 'Classification', 'value': 'clf'},
                                {'label': 'Regression', 'value': 'reg'},
                            ],
                            style={'width': '100%'}
                        ),
                    ], style={'display': 'flex'}),
                    html.Div(style={'height': '10px'}),
                    html.Div(id='evo-search-inputs-container'),
                    html.Div(style={'height': '10px'}),
                    html.Div(id='evo-search-plot-container'),
                ],
            ),
            dbc.AccordionItem(
                title='Hyperopt',
                children=[
                    html.Div([
                        html.B('Model Type', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Dropdown(
                            id='hyperopt-model-type',
                            options=[
                                {'label': 'Classification', 'value': 'clf'},
                                {'label': 'Regression', 'value': 'reg'},
                            ],
                            style={'width': '100%'}
                        ),
                    ], style={'display': 'flex'}),
                    html.Div(style={'height': '10px'}),
                    html.Div(id='hyperopt-inputs-container'),
                    html.Div(style={'height': '10px'}),
                    html.Div(id='hyperopt-plot-container'),
                ],
            ),
            dbc.AccordionItem(
                title='Optuna',
                children=[
                    html.Div([
                        html.B('Model Type', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Dropdown(
                            id='optuna-model-type',
                            options=[
                                {'label': 'Classification', 'value': 'clf'},
                                {'label': 'Regression', 'value': 'reg'},
                            ],
                            style={'width': '100%'}
                        ),
                    ], style={'display': 'flex'}),
                    html.Div(style={'height': '10px'}),
                    html.Div(id='optuna-inputs-container'),
                    html.Div(style={'height': '10px'}),
                    html.Div(id='optuna-plot-container'),
                ],
            ),
        ], start_collapsed=True, always_open=False),
    ],
    id='hyperparameter-tuning-container',
    fluid=True,
    style={
        'border-style': 'solid',
        'border-color': 'grey',
        'border-width': '1px',
        'padding': '12px',
    }
    ),
    dbc.Popover(
        [],
        target='hyperparameter-tuning-container',
        trigger='hover',
        hide_arrow=True,
        id='hyperparameter-tuning-container-popover',
    ),
]
modeling = [
    html.H2('Modeling'),
    dmc.Container([
        dbc.Accordion([
            dbc.AccordionItem(
                title='Classification',
                children=[
                    html.Div([
                        html.B('Target', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Dropdown(
                            id='clf-target',
                            options=[],
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '20px'}),
                        html.B('Features', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Dropdown(
                            id='clf-features',
                            options=[],
                            multi=True,
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '20px'}),
                        html.B('Estimators', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Dropdown(
                            id='clf-estimators',
                            options=[
                                {'label': 'Logistic Regression', 'value': 'lr_clf'},
                                {'label': 'Random Forest', 'value': 'rf_clf'},
                                {'label': 'XGBoost', 'value': 'xgb_clf'},
                            ],
                            multi=True,
                            style={'width': '100%'}
                        ),
                    ], style={'display': 'flex'}),
                    html.Div(style={'height': '10px'}),
                    html.Div([
                        html.B('CV', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Dropdown(
                            id='clf-cv',
                            options=[
                                {'label': 'KFold', 'value': 'kfold'},
                                {'label': 'Stratified K-Fold', 'value': 'skfold'},
                                {'label': 'Time Series Split', 'value': 'tssplit'},
                                {'label': 'CPCV', 'value': 'cpcv'},
                            ],
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '20px'}),
                        html.B('Probability Calibration', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Dropdown(
                            id='calibration-method',
                            options=[
                                {'label': 'None', 'value': 'none'},
                                {'label': 'Sigmoid', 'value': 'sigmoid'},
                                {'label': 'Isotonic', 'value': 'isotonic'},
                            ],
                            style={'width': '100%'}
                        ),
                    ], style={'display': 'flex'}),
                    html.Div(style={'height': '10px'}),
                    html.Div(id='clf-hyperparam-inputs-container'),
                    html.Hr(),
                    dbc.Button(
                        'Build Model',
                        id='build-clf-model',
                        color='primary',
                        style={'width': '100%'}
                    ),
                    html.Hr(),
                    html.Div(id='clf-outputs-container'),
                ]
            ),
            dbc.AccordionItem(
                title='Regression',
                children=[
                    html.Div([
                        html.B('Target', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Dropdown(
                            id='reg-target',
                            options=[],
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '20px'}),
                        html.B('Features', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Dropdown(
                            id='reg-features',
                            options=[],
                            multi=True,
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '20px'}),
                    ], style={'display': 'flex'}),
                    html.Div(style={'height': '10px'}),
                    html.Div([
                        html.B('Estimators', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Dropdown(
                            id='reg-estimators',
                            options=[
                                {'label': 'Linear Regression', 'value': 'lr_reg'},
                                {'label': 'Random Forest', 'value': 'rf_reg'},
                                {'label': 'XGBoost', 'value': 'xgb_reg'},
                            ],
                            multi=True,
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '20px'}),
                        html.B('CV', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Dropdown(
                            id='reg-cv',
                            options=[
                                {'label': 'KFold', 'value': 'kfold'},
                                {'label': 'Time Series Split', 'value': 'tssplit'},
                                {'label': 'CPCV', 'value': 'cpcv'},
                            ],
                            style={'width': '100%'}
                        ),
                    ], style={'display': 'flex'}),
                    html.Div(style={'height': '10px'}),
                    html.Div(id='reg-hyperparam-inputs-container'),
                    html.Hr(),
                    dbc.Button(
                        'Build Model',
                        id='build-reg-model',
                        color='primary',
                        style={'width': '100%'}
                    ),
                    html.Hr(),
                    html.Div(id='reg-outputs-container'),
                ]
            ),
        ], start_collapsed=True, always_open=False),
    ],
    id='modeling-container',
    fluid=True,
    style={
        'border-style': 'solid',
        'border-color': 'grey',
        'border-width': '1px',
        'padding': '12px',
    }
    ),
    dbc.Popover(
        [],
        target='modeling-container',
        trigger='hover',
        hide_arrow=True,
        id='modeling-container-popover',
    ),
]
forecasting = [
    html.H2('Forecasting'),
    dmc.Container([
        dbc.Accordion([
            dbc.AccordionItem(
                title='Forecasts',
                children=[
                    html.Div([
                        html.B('Pre-compiled Models', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                        html.Div(style={'width': '10px'}),
                        dcc.Dropdown(
                            id='pre-compiled-models',
                            options=get_saved_models(),
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '20px'}),
                        dbc.Button(
                            '🗑️ Delete Model',
                            id='delete-model',
                            color='primary',
                            style={'width': '100%'}
                        ),
                        html.Div(style={'width': '20px'}),
                        dbc.Button(
                            '🚀 Generate Forecast',
                            id='generate-forecast',
                            color='primary',
                            style={'width': '100%'}
                        ),
                    ], style={'display': 'flex'}),
                    html.Div(style={'height': '10px'}),
                    html.Div(id='forecast-features-container'),
                    html.Div(style={'height': '10px'}),
                    html.Div(id='forecast-results')
                ]
            ),
        ], start_collapsed=True, always_open=False),
    ],
    id='forecasting-container',
    fluid=True,
    style={
        'border-style': 'solid',
        'border-color': 'grey',
        'border-width': '1px',
        'padding': '12px',
    }
    ),
    dbc.Popover(
        [],
        target='forecasting-container',
        trigger='hover',
        hide_arrow=True,
        id='forecasting-container-popover',
    ),
]

##################
# Initialize App #
##################
curr_dir = os.path.dirname(os.path.abspath(__file__))
dbc_css = ("https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.2/dbc.min.css")
external_stylesheets = [dbc.themes.DARKLY, dbc_css]
app = DashProxy(
    __name__,
    external_stylesheets=external_stylesheets,
    prevent_initial_callbacks=True,
    suppress_callback_exceptions=True,
    transforms=[MultiplexerTransform()],
    title='Quant Toolkit'
)
app.layout = html.Div([
    dcc.Location(id="url"),

    # storage and download
    dcc.Store(id='the-data'),
    dcc.Store(id='hyperparameter-search-results'),
    dcc.Store(id='clf-results'),
    dcc.Store(id='reg-results'),
    dcc.Download(id="to-csv"),

    # sidebar
    html.Div([
        html.Img(src=app.get_asset_url('app_logo.png'), style={'width': '100%'}),
        dbc.Nav(
            [
                ############################################################################################
                # DATA
                html.Div([
                    html.H4(
                        style={
                            'display': 'flex',
                            'flex-direction': 'row',
                            'content': '',
                            'flex': '1 1',
                            'border-bottom': '1px solid #FFFFFF',
                            'margin-right': '0.5rem',
                        }
                    ),
                    html.B('Data', style={'margin-bottom': '10px', 'white-space': 'nowrap'}),
                    html.H4(
                        style={
                            'display': 'flex',
                            'flex-direction': 'row',
                            'content': '',
                            'flex': '1 1',
                            'border-bottom': '1px solid #FFFFFF',
                            'margin-left': '0.5rem',
                        }
                    ),
                ], style={'display': 'flex', 'flex-direction': 'row', 'align-items': 'center', 'margin-top': '10px'}),
                # add data
                dbc.Button(
                    html.A([
                        DashIconify(
                            icon="bi:database-fill-add",
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Add Data",
                    ],
                        href='#add-data',
                        style={'color': 'white', 'text-decoration': 'none'}
                    ),
                    outline=True,
                    style={'textAlign': 'left', 'width': '100%'},
                    id='navbutton-add-data'
                ),
                dbc.Popover(
                    [],
                    target='navbutton-add-data',
                    trigger='hover',
                    hide_arrow=True,
                    id='add-data-navbutton-popover',
                ),
                # add features
                dbc.Button(
                    html.A([
                        DashIconify(
                            icon="material-symbols:add-chart-rounded",
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Add Features",
                    ], href='#add-features', style={'color': 'white', 'text-decoration': 'none'}),
                    outline=True,
                    style={'textAlign': 'left', 'width': '100%'},
                    id='navbutton-add-features'
                ),
                dbc.Popover(
                    [],
                    target='navbutton-add-features',
                    trigger='hover',
                    hide_arrow=True,
                    id='add-features-navbutton-popover',
                ),
                # preprocessing
                dbc.Button(
                    html.A([
                        DashIconify(
                            icon="carbon:clean",
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Preprocessing",
                    ], href='#preprocessing', style={'color': 'white', 'text-decoration': 'none'}),
                    outline=True,
                    style={'textAlign': 'left', 'width': '100%'},
                    id='navbutton-preprocessing'
                ),
                dbc.Popover(
                    [],
                    target='navbutton-preprocessing',
                    trigger='hover',
                    hide_arrow=True,
                    id='preprocessing-navbutton-popover',
                ),
                ############################################################################################
                # ANALYSIS
                html.Div([
                    html.H4(
                        style={
                            'display': 'flex',
                            'flex-direction': 'row',
                            'content': '',
                            'flex': '1 1',
                            'border-bottom': '1px solid #FFFFFF',
                            'margin-right': '0.5rem',
                        }
                    ),
                    html.B('Analysis Tools', style={'margin-bottom': '10px', 'white-space': 'nowrap'}),
                    html.H4(
                        style={
                            'display': 'flex',
                            'flex-direction': 'row',
                            'content': '',
                            'flex': '1 1',
                            'border-bottom': '1px solid #FFFFFF',
                            'margin-left': '0.5rem',
                        }
                    ),
                ], style={'display': 'flex', 'flex-direction': 'row', 'align-items': 'center', 'margin-top': '10px'}),
                # distributions
                dbc.Button(
                    html.A([
                        DashIconify(
                            icon="mdi:chart-bell-curve",
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Distributions",
                    ], href='#distributions', style={'color': 'white', 'text-decoration': 'none'}),
                    outline=True,
                    style={'textAlign': 'left', 'width': '100%'},
                    id='navbutton-distributions'
                ),
                dbc.Popover(
                    [],
                    target='navbutton-distributions',
                    trigger='hover',
                    hide_arrow=True,
                    id='distributions-navbutton-popover',
                ),
                # statistical association
                dbc.Button(
                    html.A([
                        DashIconify(
                            icon="fluent:arrow-trending-lines-20-filled",
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Statistical Association",
                    ], href='#statistical-association', style={'color': 'white', 'text-decoration': 'none'}),
                    outline=True,
                    style={'textAlign': 'left', 'width': '100%'},
                    id='navbutton-statistical-association'
                ),
                dbc.Popover(
                    [],
                    target='navbutton-statistical-association',
                    trigger='hover',
                    hide_arrow=True,
                    id='statistical-association-navbutton-popover',
                ),
                # temporal sequence
                dbc.Button(
                    html.A([
                        DashIconify(
                            icon="ic:outline-access-time",
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Temporal Sequence",
                    ], href='#temporal-sequence', style={'color': 'white', 'text-decoration': 'none'}),
                    outline=True,
                    style={'textAlign': 'left', 'width': '100%'},
                    id='navbutton-temporal-sequence'
                ),
                dbc.Popover(
                    [],
                    target='navbutton-temporal-sequence',
                    trigger='hover',
                    hide_arrow=True,
                    id='temporal-sequence-navbutton-popover',
                ),
                ############################################################################################
                # MACHINE LEARNING
                html.Div([
                    html.H4(
                        style={
                            'display': 'flex',
                            'flex-direction': 'row',
                            'content': '',
                            'flex': '1 1',
                            'border-bottom': '1px solid #FFFFFF',
                            'margin-right': '0.5rem',
                        }
                    ),
                    html.B('Machine Learning', style={'margin-bottom': '10px', 'white-space': 'nowrap'}),
                    html.H4(
                        style={
                            'display': 'flex',
                            'flex-direction': 'row',
                            'content': '',
                            'flex': '1 1',
                            'border-bottom': '1px solid #FFFFFF',
                            'margin-left': '0.5rem',
                        }
                    ),
                ], style={'display': 'flex', 'flex-direction': 'row', 'align-items': 'center', 'margin-top': '10px'}),             
                # feature importance
                dbc.Button(
                    html.A([
                        DashIconify(
                            icon="healthicons:medium-bars",
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Feature Importance",
                    ], href='#feature-importance', style={'color': 'white', 'text-decoration': 'none'}),
                    outline=True,
                    style={'textAlign': 'left', 'width': '100%'},
                    id='navbutton-feature-importance'
                ),
                dbc.Popover(
                    [],
                    target='navbutton-feature-importance',
                    trigger='hover',
                    hide_arrow=True,
                    id='feature-importance-navbutton-popover',
                ),
                # dimensionality reduction
                dbc.Button(
                    html.A([
                        DashIconify(
                            icon="fluent-mdl2:web-components",
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Dimensionality Reduction",
                    ], href='#dimensionality-reduction', style={'color': 'white', 'text-decoration': 'none'}),
                    outline=True,
                    style={'textAlign': 'left', 'width': '100%'},
                    id='navbutton-dimensionality-reduction'
                ),
                dbc.Popover(
                    [],
                    target='navbutton-dimensionality-reduction',
                    trigger='hover',
                    hide_arrow=True,
                    id='dimensionality-reduction-navbutton-popover',
                ),
                # hyperparameter tuning
                dbc.Button(
                    html.A([
                        DashIconify(
                            icon="zondicons:tuning",
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Hyperparameter Tuning",
                    ], href='#hyperparameter-tuning', style={'color': 'white', 'text-decoration': 'none'}),
                    outline=True,
                    style={'textAlign': 'left', 'width': '100%'},
                    id='navbutton-hyperparameter-tuning'
                ),
                dbc.Popover(
                    [],
                    target='navbutton-hyperparameter-tuning',
                    trigger='hover',
                    hide_arrow=True,
                    id='hyperparameter-tuning-navbutton-popover',
                ),
                # modeling
                dbc.Button(
                    html.A([
                        DashIconify(
                            icon="fluent-mdl2:server-processes",
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Modeling",
                    ], href='#modeling', style={'color': 'white', 'text-decoration': 'none'}),
                    outline=True,
                    style={'textAlign': 'left', 'width': '100%'},
                    id='navbutton-modeling'
                ),
                dbc.Popover(
                    [],
                    target='navbutton-modeling',
                    trigger='hover',
                    hide_arrow=True,
                    id='modeling-navbutton-popover',
                ),
                # forecasting
                dbc.Button(
                    html.A([
                        DashIconify(
                            icon="carbon:forecast-lightning",
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Forecasting",
                    ], href='#forecasting', style={'color': 'white', 'text-decoration': 'none'}),
                    outline=True,
                    style={'textAlign': 'left', 'width': '100%'},
                    id='navbutton-forecasting'
                ),
                dbc.Popover(
                    [],
                    target='navbutton-forecasting',
                    trigger='hover',
                    hide_arrow=True,
                    id='forecasting-navbutton-popover',
                ),
                ############################################################################################
                # ALERTS
                dbc.Alert(
                    '✔ SUCCESS!',
                    id='success-alert',
                    is_open=False,
                    dismissable=False,
                    duration=3000,
                    color='#90da86',
                    style={
                        'text-align': 'center',
                        'font-weight': 'bold',
                        'font-size': '25px',
                        'color': '#1f351c',
                        'opacity': '0.75'
                    }
                )
            ],
            vertical=True,
            pills=True,
            navbar=True,
            navbar_scroll=True,
        )], style={
        "overflow": "scroll",
        "position": "fixed",
        'text-align': 'left',
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "21rem",
        "padding": "1rem 1rem",
        "background-color": "#5A5A5A",
    }),

    # page content
    dmc.Timeline(
        id='timeline',
        lineWidth=2,
        bulletSize=35,
        children=[
            # Data
            dmc.TimelineItem(
                id='add-data',
                lineVariant='solid',
                children=add_data,
                bullet=DashIconify(icon="bi:database-fill-add", width=25, height=25),
                style={'color': '#FFFFFF'},
            ),
            dmc.TimelineItem(
                id='add-features',
                lineVariant='solid',
                children=add_features,
                bullet=DashIconify(icon="material-symbols:add-chart-rounded", width=25, height=25),
                style={'color': '#FFFFFF'},
            ),
            dmc.TimelineItem(
                id='preprocessing',
                lineVariant='solid',
                children=preprocessing,
                bullet=DashIconify(icon="carbon:clean", width=25, height=25),
                style={'color': '#FFFFFF'},
            ),
            # Analysis Tools
            dmc.TimelineItem(
                id='distributions',
                lineVariant='solid',
                children=distributions,
                bullet=DashIconify(icon="mdi:chart-bell-curve", width=25, height=25),
                style={'color': '#FFFFFF'},
            ),
            dmc.TimelineItem(
                id='statistical-association',
                lineVariant='solid',
                children=statistical_association,
                bullet=DashIconify(icon="fluent:arrow-trending-lines-20-filled", width=25, height=25),
                style={'color': '#FFFFFF'},
            ),
            dmc.TimelineItem(
                id='temporal-sequence',
                lineVariant='solid',
                children=temporal_sequence,
                bullet=DashIconify(icon="ic:outline-access-time", width=25, height=25),
                style={'color': '#FFFFFF'},
            ),
            # Machine Learning
            dmc.TimelineItem(
                id='feature-importance',
                lineVariant='solid',
                children=feature_importance,
                bullet=DashIconify(icon="healthicons:medium-bars", width=25, height=25),
                style={'color': '#FFFFFF'},
            ),
            dmc.TimelineItem(
                id='dimensionality-reduction',
                lineVariant='solid',
                children=dimensionality_reduction,
                bullet=DashIconify(icon="fluent-mdl2:web-components", width=25, height=25),
                style={'color': '#FFFFFF'},
            ),
            dmc.TimelineItem(
                id='hyperparameter-tuning',
                lineVariant='solid',
                children=hyperparameter_tuning,
                bullet=DashIconify(icon="zondicons:tuning", width=25, height=25),
                style={'color': '#FFFFFF'},
            ),
            dmc.TimelineItem(
                id='modeling',
                lineVariant='solid',
                children=modeling,
                bullet=DashIconify(icon="fluent-mdl2:server-processes", width=25, height=25),
                style={'color': '#FFFFFF'},
            ),
            dmc.TimelineItem(
                id='forecasting',
                lineVariant='solid',
                children=forecasting,
                bullet=DashIconify(icon="carbon:forecast-lightning", width=25, height=25),
                style={'color': '#FFFFFF'},
            ),
        ],
        style={
            'margin-left': '22rem',
            'margin-right': '1rem',
            "padding": "1rem 1rem",
            "content": "width=device-width, initial-scale=1, maximum-scale=1",
        }
    ),

    # hidden divs (for callbacks that don't return anything)
    html.Div(id='hidden-div', style={'display':'none'}),

], className="dbc")


#######################################
# Update UI Components on Data Change #
####################################### 
@app.callback(
    # Table
    Output('datatable', 'data'),
    Output('datatable', 'columns'),
    # Input Components
    Output('drop-features-dropdown', 'options'),
    Output('rename-features-dropdown', 'options'),
    Output('type-casting-feature-dropdown', 'options'),
    Output('order-by-feature-dropdown', 'options'),
    Output('transformations-base-features', 'options'),
    Output('open-feature-dropdown', 'options'),
    Output('high-feature-dropdown', 'options'),
    Output('low-feature-dropdown', 'options'),
    Output('close-feature-dropdown', 'options'),
    Output('volume-feature-dropdown', 'options'),
    Output('var-plot-sliders', 'options'),
    Output('dist-plot-feature', 'options'),
    Output('dist-plot-sliders', 'options'),
    Output('line-plot-features', 'options'),
    Output('assoc-matrix-sliders', 'options'),
    Output('joint-plot-feature-x', 'options'),
    Output('joint-plot-feature-y', 'options'),
    Output('joint-plot-sliders', 'options'),
    Output('heatmap-feature-x', 'options'),
    Output('heatmap-feature-y', 'options'),
    Output('heatmap-magnitude', 'options'),
    Output('heatmap-sliders', 'options'),
    Output('drift-plot-feature', 'options'),
    Output('drift-plot-n-splits', 'max'),
    Output('boruta-shap-target', 'options'),
    Output('clf-target', 'options'),
    Output('clf-features', 'options'),
    Output('reg-target', 'options'),
    Output('reg-features', 'options'),
    # Plots
    Output('missing-values-plot', 'figure'),
    # Inputs
    Input('hidden-div', 'children'),
    State('the-data', 'data'),
)
def update_ui_components(_, data):
    if data:
        df = pd.DataFrame.from_dict(data)
        missing_values = df.isnull().sum()
        features = [{"label": i, "value": i} for i in df.columns]
        return [
            # Table
            data,
            [{"name": i, "id": i} for i in df.columns],
            # Input Components
            features, # drop-features-dropdown
            features, # rename-features-dropdown
            features, # type-casting-feature-dropdown
            [{'label': 'Index', 'value': 'index'}] + features, # order-by-feature-dropdown
            features, # transformations-base-features
            features, # open-feature-dropdown
            features, # high-feature-dropdown
            features, # low-feature-dropdown
            features, # close-feature-dropdown
            features, # volume-feature-dropdown
            [{'label': 'Index', 'value': 'index'}] + features, # var-plot-sliders
            features, # dist-plot-feature
            [{'label': 'Index', 'value': 'index'}] + features, # dist-plot-sliders
            features, # line-plot-features
            [{'label': 'Index', 'value': 'index'}] + features, # assoc-matrix-sliders
            features, # joint-plot-feature-x
            features, # joint-plot-feature-y
            [{'label': 'Index', 'value': 'index'}] + features, # joint-plot-sliders
            features, # heatmap-feature-x
            features, # heatmap-feature-y
            [{'label': 'Density', 'value': 'density'}] + features, # heatmap-magnitude
            [{'label': 'Index', 'value': 'index'}] + features, # heatmap-sliders
            features, # drift-plot-feature
            math.floor(len(df.index) / 2), # drift-plot-n-splits
            features, # boruta-shap-target
            features, # clf-target
            [{'label': 'ALL FEATURES', 'value': 'ALL FEATURES'}] + features, # clf-features
            features, # reg-target
            [{'label': 'ALL FEATURES', 'value': 'ALL FEATURES'}] + features, # reg-features
            # Plots
            go.Figure(
                data=[
                    go.Bar(
                        x=missing_values.index,
                        y=missing_values.values,
                        name='Number of Nulls',
                        marker=dict(color='#37699b'),
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

###########################
# Navbutton Hover Lighten #
###########################
@app.callback(
    Output('navbutton-add-data', 'outline'),
    Output('add-data', 'lineVariant'),
    Input('add-data-navbutton-popover', 'is_open'),
    Input('add-data-container-popover', 'is_open')
)
def hover_add_data(is_open1, is_open2):
    if is_open1 or is_open2:
        return False, 'dashed'
    return True, 'solid'

@app.callback(
    Output('navbutton-add-features', 'outline'),
    Output('add-features', 'lineVariant'),
    Input('add-features-navbutton-popover', 'is_open'),
    Input('add-features-container-popover', 'is_open')
)
def hover_add_features(is_open1, is_open2):
    if is_open1 or is_open2:
        return False, 'dashed'
    return True, 'solid'

@app.callback(
    Output('navbutton-preprocessing', 'outline'),
    Output('preprocessing', 'lineVariant'),
    Input('preprocessing-navbutton-popover', 'is_open'),
    Input('preprocessing-container-popover', 'is_open')
)
def hover_preprocessing(is_open1, is_open2):
    if is_open1 or is_open2:
        return False, 'dashed'
    return True, 'solid'

@app.callback(
    Output('navbutton-price-charts', 'outline'),
    Output('price-charts', 'lineVariant'),
    Input('price-charts-navbutton-popover', 'is_open'),
    Input('price-charts-container-popover', 'is_open')
)
def hover_price_charts(is_open1, is_open2):
    if is_open1 or is_open2:
        return False, 'dashed'
    return True, 'solid'

@app.callback(
    Output('navbutton-distributions', 'outline'),
    Output('distributions', 'lineVariant'),
    Input('distributions-navbutton-popover', 'is_open'),
    Input('distributions-container-popover', 'is_open')
)
def hover_distributions(is_open1, is_open2):
    if is_open1 or is_open2:
        return False, 'dashed'
    return True, 'solid'

@app.callback(
    Output('navbutton-statistical-association', 'outline'),
    Output('statistical-association', 'lineVariant'),
    Input('statistical-association-navbutton-popover', 'is_open'),
    Input('statistical-association-container-popover', 'is_open')
)
def hover_statistical_association(is_open1, is_open2):
    if is_open1 or is_open2:
        return False, 'dashed'
    return True, 'solid'

@app.callback(
    Output('navbutton-temporal-sequence', 'outline'),
    Output('temporal-sequence', 'lineVariant'),
    Input('temporal-sequence-navbutton-popover', 'is_open'),
    Input('temporal-sequence-container-popover', 'is_open')
)
def hover_temporal_sequence(is_open1, is_open2):
    if is_open1 or is_open2:
        return False, 'dashed'
    return True, 'solid'

@app.callback(
    Output('navbutton-feature-importance', 'outline'),
    Output('feature-importance', 'lineVariant'),
    Input('feature-importance-navbutton-popover', 'is_open'),
    Input('feature-importance-container-popover', 'is_open')
)
def hover_feature_importance(is_open1, is_open2):
    if is_open1 or is_open2:
        return False, 'dashed'
    return True, 'solid'

@app.callback(
    Output('navbutton-dimensionality-reduction', 'outline'),
    Output('dimensionality-reduction', 'lineVariant'),
    Input('dimensionality-reduction-navbutton-popover', 'is_open'),
    Input('dimensionality-reduction-container-popover', 'is_open')
)
def hover_dimensionality_reduction(is_open1, is_open2):
    if is_open1 or is_open2:
        return False, 'dashed'
    return True, 'solid'

@app.callback(
    Output('navbutton-hyperparameter-tuning', 'outline'),
    Output('hyperparameter-tuning', 'lineVariant'),
    Input('hyperparameter-tuning-navbutton-popover', 'is_open'),
    Input('hyperparameter-tuning-container-popover', 'is_open')
)
def hover_hyperparameter_tuning(is_open1, is_open2):
    if is_open1 or is_open2:
        return False, 'dashed'
    return True, 'solid'

@app.callback(
    Output('navbutton-modeling', 'outline'),
    Output('modeling', 'lineVariant'),
    Input('modeling-navbutton-popover', 'is_open'),
    Input('modeling-container-popover', 'is_open')
)
def hover_modeling(is_open1, is_open2):
    if is_open1 or is_open2:
        return False, 'dashed'
    return True, 'solid'

@app.callback(
    Output('navbutton-forecasting', 'outline'),
    Output('forecasting', 'lineVariant'),
    Input('forecasting-navbutton-popover', 'is_open'),
    Input('forecasting-container-popover', 'is_open')
)
def hover_forecasting(is_open1, is_open2):
    if is_open1 or is_open2:
        return False, 'dashed'
    return True, 'solid'
