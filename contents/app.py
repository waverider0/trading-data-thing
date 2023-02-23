import os
import pandas as pd

from dash import html, dcc
from dash_extensions.enrich import DashProxy, MultiplexerTransform
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from dash import dash_table
import dash_bootstrap_components as dbc
import dash_loading_spinners as dls
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import plotly.graph_objs as go


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
        hide_arrow=False,
        id='add-data-container-popover',
    ),
    html.Div(style={'height': '10px'}),
]
add_features = [
    html.H2('Add Features'),
    dmc.Container([
        dbc.Accordion([
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
                                {'label': 'Lag', 'value': 'lag'},
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
            ),
            dbc.AccordionItem(
                title="Filters",
            ),
            dbc.AccordionItem(
                title="Metalabeling",
            ),
        ], start_collapsed=True, always_open=False),
    ], 
    fluid=True,
    style={
        'border-style': 'solid',
        'border-color': 'grey',
        'border-width': '1px',
        'padding': '12px',
    }
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
    fluid=True,
    style={
        'border-style': 'solid',
        'border-color': 'grey',
        'border-width': '1px',
        'padding': '12px',
    }
    ),
]

price_charts = [
    html.H2('Price Charts'),
    dmc.Container([
        dbc.Accordion([
            dbc.AccordionItem(
                title="Candlestick",
            ),
            dbc.AccordionItem(
                title="Volume Profile",
            ),
            dbc.AccordionItem(
                title="Time Price Opportunity",
            ),
        ], start_collapsed=True, always_open=False),
    ],
    fluid=True,
    style={
        'border-style': 'solid',
        'border-color': 'grey',
        'border-width': '1px',
        'padding': '12px',
    }
    )
]
distributions = [
    html.H2('Distributions'),
    dmc.Container([
        dbc.Accordion([
            dbc.AccordionItem(
                title="Variance Plots",
            ),
            dbc.AccordionItem(
                title="Distribution Plots",
            ),
            dbc.AccordionItem(
                title="QQ Plots",
            ),
            dbc.AccordionItem(
                title="Normality Tests",
            ),
        ], start_collapsed=True, always_open=False),
    ],
    fluid=True,
    style={
        'border-style': 'solid',
        'border-color': 'grey',
        'border-width': '1px',
        'padding': '12px',
    }
    ),
]
statistical_association = [
    html.H2('Statistical Association'),
    dmc.Container([
        dbc.Accordion([
            dbc.AccordionItem(
                title="Line Plots",
            ),
            dbc.AccordionItem(
                title="Association Matrices",
            ),
            dbc.AccordionItem(
                title="Heatmaps",
            ),
            dbc.AccordionItem(
                title="Joint Plots",
            ),
        ], start_collapsed=True, always_open=False),
    ],
    fluid=True,
    style={
        'border-style': 'solid',
        'border-color': 'grey',
        'border-width': '1px',
        'padding': '12px',
    }
    ),

]
temporal_sequence = [
    html.H2('Temporal Sequence'),
    dmc.Container([
        dbc.Accordion([
            dbc.AccordionItem(
                title='Dynamic Time Warping',
            ),
        ], start_collapsed=True, always_open=False),
    ],
    fluid=True,
    style={
        'border-style': 'solid',
        'border-color': 'grey',
        'border-width': '1px',
        'padding': '12px',
    }
    ),
]

feature_importance = [
    html.H2('Feature Importance'),
    dmc.Container([
        dbc.Accordion([
            dbc.AccordionItem(
                title='Recursive Feature Elimination',
            ),
            dbc.AccordionItem(
                title='Sequential Feature Selection',
            ),
            dbc.AccordionItem(
                title='Boruta SHAP',
            ),
        ], start_collapsed=True, always_open=False),
    ],
    fluid=True,
    style={
        'border-style': 'solid',
        'border-color': 'grey',
        'border-width': '1px',
        'padding': '12px',
    }
    ),
]
dimensionality_reduction = [
    html.H2('Dimensionality Reduction'),
    dmc.Container([
        dbc.Accordion([
            dbc.AccordionItem(
                title='Matrix Decomposition',
            ),
            dbc.AccordionItem(
                title='Manifold Learning',
            ),
            dbc.AccordionItem(
                title='Discriminant Analysis',
            ),
        ], start_collapsed=True, always_open=False),
    ],
    fluid=True,
    style={
        'border-style': 'solid',
        'border-color': 'grey',
        'border-width': '1px',
        'padding': '12px',
    }
    ),
]
hyperparameter_tuning = [
    html.H2('Hyperparameter Tuning'),
    dmc.Container([
        dbc.Accordion([
            dbc.AccordionItem(
                title='Grid Search',
            ),
            dbc.AccordionItem(
                title='Random Search',
            ),
            dbc.AccordionItem(
                title='Bayesian Search',
            ),
            dbc.AccordionItem(
                title='Evolutionary Search',
            ),
            dbc.AccordionItem(
                title='Hyperopt',
            ),
            dbc.AccordionItem(
                title='Optuna',
            ),
        ], start_collapsed=True, always_open=False),
    ],
    fluid=True,
    style={
        'border-style': 'solid',
        'border-color': 'grey',
        'border-width': '1px',
        'padding': '12px',
    }
    ),
]
modeling = [
    html.H2('Modeling'),
    dmc.Container([
        dbc.Accordion([
            dbc.AccordionItem(
                title='Classification',
            ),
            dbc.AccordionItem(
                title='Regression',
            ),
        ], start_collapsed=True, always_open=False),
    ],
    fluid=True,
    style={
        'border-style': 'solid',
        'border-color': 'grey',
        'border-width': '1px',
        'padding': '12px',
    }
    ),
]
forecasting = [
    html.H2('Forecasting'),
    dmc.Container([
        dbc.Accordion([
            dbc.AccordionItem(
                title='Forecasts',
            ),
        ], start_collapsed=True, always_open=False),
    ],
    fluid=True,
    style={
        'border-style': 'solid',
        'border-color': 'grey',
        'border-width': '1px',
        'padding': '12px',
    }
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
    suppress_callback_exceptions=True,
    prevent_initial_callbacks=True,
    transforms=[MultiplexerTransform()],
    title='Quant Toolkit'
)
app.layout = html.Div([
    dcc.Location(id="url"),

    # storage and download
    dcc.Store(id='the-data'),
    dcc.Store(id='hyperparameter-search-results'),
    dcc.Store(id='raw-modeling-results'),
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
                    id='add-data-popover',
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
                    id='add-features-popover',
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
                    id='preprocessing-popover',
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
                # price charts
                dbc.Button(
                    html.A([
                        DashIconify(
                            icon="iconoir:candlestick-chart",
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Price Charts",
                    ], href='#price-charts', style={'color': 'white', 'text-decoration': 'none'}),
                    outline=True,
                    style={'textAlign': 'left', 'width': '100%'},
                    id='navbutton-price-charts'
                ),
                dbc.Popover(
                    [],
                    target='navbutton-price-charts',
                    trigger='hover',
                    hide_arrow=True,
                    id='price-charts-popover',
                ),
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
                    id='distributions-popover',
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
                    id='statistical-association-popover',
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
                    id='temporal-sequence-popover',
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
                    id='feature-importance-popover',
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
                    id='dimensionality-reduction-popover',
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
                    id='hyperparameter-tuning-popover',
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
                    id='modeling-popover',
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
                    id='forecasting-popover',
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
        #active=0,
        #color='gray',
        lineWidth=2,
        bulletSize=35,
        children=[
            # Data
            dmc.TimelineItem(
                id='add-data',
                children=add_data,
                bullet=DashIconify(icon="bi:database-fill-add", width=25, height=25),
                style={'color': '#FFFFFF'},
            ),
            dmc.TimelineItem(
                id='add-features',
                children=add_features,
                bullet=DashIconify(icon="material-symbols:add-chart-rounded", width=25, height=25),
                style={'color': '#FFFFFF'},
            ),
            dmc.TimelineItem(
                id='preprocessing',
                children=preprocessing,
                bullet=DashIconify(icon="carbon:clean", width=25, height=25),
                style={'color': '#FFFFFF'},
            ),
            # Analysis Tools
            dmc.TimelineItem(
                id='price-charts',
                children=price_charts,
                bullet=DashIconify(icon="iconoir:candlestick-chart", width=25, height=25),
                style={'color': '#FFFFFF'},
            ),
            dmc.TimelineItem(
                id='distributions',
                children=distributions,
                bullet=DashIconify(icon="mdi:chart-bell-curve", width=25, height=25),
                style={'color': '#FFFFFF'},
            ),
            dmc.TimelineItem(
                id='statistical-association',
                children=statistical_association,
                bullet=DashIconify(icon="fluent:arrow-trending-lines-20-filled", width=25, height=25),
                style={'color': '#FFFFFF'},
            ),
            dmc.TimelineItem(
                id='temporal-sequence',
                children=temporal_sequence,
                bullet=DashIconify(icon="ic:outline-access-time", width=25, height=25),
                style={'color': '#FFFFFF'},
            ),
            # Machine Learning
            dmc.TimelineItem(
                id='feature-importance',
                children=feature_importance,
                bullet=DashIconify(icon="healthicons:medium-bars", width=25, height=25),
                style={'color': '#FFFFFF'},
            ),
            dmc.TimelineItem(
                id='dimensionality-reduction',
                children=dimensionality_reduction,
                bullet=DashIconify(icon="fluent-mdl2:web-components", width=25, height=25),
                style={'color': '#FFFFFF'},
            ),
            dmc.TimelineItem(
                id='hyperparameter-tuning',
                children=hyperparameter_tuning,
                bullet=DashIconify(icon="zondicons:tuning", width=25, height=25),
                style={'color': '#FFFFFF'},
            ),
            dmc.TimelineItem(
                id='modeling',
                children=modeling,
                bullet=DashIconify(icon="fluent-mdl2:server-processes", width=25, height=25),
                style={'color': '#FFFFFF'},
            ),
            dmc.TimelineItem(
                id='forecasting',
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
    # Feature Dropdowns
    Output('type-casting-feature-dropdown', 'options'),
    Output('drop-features-dropdown', 'options'),
    Output('transformations-base-features', 'options'),
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
            # Feature Dropdowns
            features,
            features,
            features,
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

###########################
# Navbutton Hover Lighten #
###########################
@app.callback(
    Output('navbutton-add-data', 'outline'),
    Input('add-data-popover', 'is_open'),
)
def hover_add_data(is_open):
    if is_open:
        return False
    return True

@app.callback(
    Output('navbutton-add-features', 'outline'),
    Input('add-features-popover', 'is_open'),
)
def hover_add_features(is_open):
    if is_open:
        return False
    return True

@app.callback(
    Output('navbutton-preprocessing', 'outline'),
    Input('preprocessing-popover', 'is_open'),
)
def hover_preprocessing(is_open):
    if is_open:
        return False
    return True

@app.callback(
    Output('navbutton-price-charts', 'outline'),
    Input('price-charts-popover', 'is_open'),
)
def hover_price_charts(is_open):
    if is_open:
        return False
    return True

@app.callback(
    Output('navbutton-distributions', 'outline'),
    Input('distributions-popover', 'is_open'),
)
def hover_distributions(is_open):
    if is_open:
        return False
    return True

@app.callback(
    Output('navbutton-statistical-association', 'outline'),
    Input('statistical-association-popover', 'is_open'),
)
def hover_statistical_association(is_open):
    if is_open:
        return False
    return True

@app.callback(
    Output('navbutton-temporal-sequence', 'outline'),
    Input('temporal-sequence-popover', 'is_open'),
)
def hover_temporal_sequence(is_open):
    if is_open:
        return False
    return True

@app.callback(
    Output('navbutton-feature-importance', 'outline'),
    Input('feature-importance-popover', 'is_open'),
)
def hover_feature_importance(is_open):
    if is_open:
        return False
    return True

@app.callback(
    Output('navbutton-dimensionality-reduction', 'outline'),
    Input('dimensionality-reduction-popover', 'is_open'),
)
def hover_dimensionality_reduction(is_open):
    if is_open:
        return False
    return True

@app.callback(
    Output('navbutton-hyperparameter-tuning', 'outline'),
    Input('hyperparameter-tuning-popover', 'is_open'),
)
def hover_hyperparameter_tuning(is_open):
    if is_open:
        return False
    return True

@app.callback(
    Output('navbutton-modeling', 'outline'),
    Input('modeling-popover', 'is_open'),
)
def hover_modeling(is_open):
    if is_open:
        return False
    return True

@app.callback(
    Output('navbutton-forecasting', 'outline'),
    Input('forecasting-popover', 'is_open'),
)
def hover_forecasting(is_open):
    if is_open:
        return False
    return True

#
app.callback(
    #Output('navbutton-add-data', 'outline'),
    Output('timeline', 'active'),
    Input('add-data-container-popover', 'is_open'),
)
def hover_add_data(is_open):
    if is_open:
        return 1
    return 0