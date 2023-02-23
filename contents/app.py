import os

from dash import html, dcc
from dash_extensions.enrich import DashProxy, MultiplexerTransform
from dash.dependencies import Input, Output

from dash import dash_table
import dash_bootstrap_components as dbc
import dash_loading_spinners as dls
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import plotly.graph_objs as go


###########
# Layouts #
###########
add_data = [
    html.H2('Add Data'),
    html.Hr(),
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
            'margin': '10px'
        },
        multiple=False
    ),
    html.Div(style={'height': '10px'}),
]
add_features = [
    html.H2('Add Features'),
    html.Hr(),
    html.Div(style={'height': '10px'}),
    # accordion
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
                        id='feature-engineering-transformations-dropdown',
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
                        id='transformation-input',
                        type='number',
                        value=1,
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '30px'}),
                    dbc.Button(
                        'Add Feature',
                        id='add-feature-button',
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
    ], start_collapsed=True),
    html.Div(style={'height': '10px'}),
]
preprocessing = [
    html.H2('Preprocessing'),
    html.Hr(),
    html.Div(style={'height': '10px'}),
    # accordion
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
    ], start_collapsed=True)
]

price_charts = [
    html.H2('Price Charts'),
    html.Hr(),
    # accordion
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
    ])
]
distributions = [
    html.H2('Distributions'),
    html.Hr(),
]
statistical_association = [
    html.H2('Statistical Association'),
    html.Hr(),
]
temporal_sequence = [
    html.H2('Temporal Sequence'),
    html.Hr(),
]

feature_importance = [
    html.H2('Feature Importance'),
    html.Hr(),
]
dimensionality_reduction = [
    html.H2('Dimensionality Reduction'),
    html.Hr(),
]
hyperparameter_tuning = [
    html.H2('Hyperparameter Tuning'),
    html.Hr(),
]
modeling = [
    html.H2('Modeling'),
    html.Hr(),
]
forecasting = [
    html.H2('Forecasting'),
    html.Hr(),
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
                            color='#59fb00',
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Add Data",
                    ], href='#add-data', style={'color': 'white', 'text-decoration': 'none'}),
                    outline=True,
                    style={
                        'textAlign': 'left',
                        'width': '100%',
                    },
                    id={
                        'type': 'navbutton',
                        'module': 'add-data',
                    }
                ),
                # add features
                dbc.Button(
                    html.A([
                        DashIconify(
                            icon="material-symbols:add-chart-rounded",
                            color='#59fb00',
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Add Features",
                    ], href='#add-features', style={'color': 'white', 'text-decoration': 'none'}),
                    outline=True,
                    style={
                        'textAlign': 'left',
                        'width': '100%',
                    },
                    id={
                        'type': 'navbutton',
                        'module': 'add-features',
                    }
                ),
                # preprocessing
                dbc.Button(
                    html.A([
                        DashIconify(
                            icon="carbon:clean",
                            color='#59fb00',
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Preprocessing",
                    ], href='#preprocessing', style={'color': 'white', 'text-decoration': 'none'}),
                    outline=True,
                    style={
                        'textAlign': 'left',
                        'width': '100%',
                    },
                    id={
                        'type': 'navbutton',
                        'module': 'preprocessing',
                    }
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
                            color='#fb7d00',
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Price Charts",
                    ], href='#price-charts', style={'color': 'white', 'text-decoration': 'none'}),
                    outline=True,
                    style={
                        'textAlign': 'left',
                        'width': '100%',
                    },
                    id={
                        'type': 'navbutton',
                        'module': 'price-charts',
                    }
                ),
                # distributions
                dbc.Button(
                    dmc.Anchor([
                        DashIconify(
                            icon="mdi:chart-bell-curve",
                            color='#fb7d00',
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Distributions",
                    ], href='#distributions', style={'color': 'white', 'text-decoration': 'none'}),
                    outline=True,
                    style={
                        'textAlign': 'left',
                        'width': '100%',
                    },
                    id={
                        'type': 'navbutton',
                        'module': 'distributions',
                    }
                ),
                # statistical association
                dbc.Button(
                    dmc.Anchor([
                        DashIconify(
                            icon="fluent:arrow-trending-lines-20-filled",
                            color='#fb7d00',
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Statistical Association",
                    ], href='#statistical-association', style={'color': 'white', 'text-decoration': 'none'}),
                    outline=True,
                    style={
                        'textAlign': 'left',
                        'width': '100%',
                    },
                    id={
                        'type': 'navbutton',
                        'module': 'statistical-association',
                    }
                ),
                # temporal sequence
                dbc.Button(
                    dmc.Anchor([
                        DashIconify(
                            icon="ic:outline-access-time",
                            color='#fb7d00',
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Temporal Sequence",
                    ], href='#temporal-sequence', style={'color': 'white', 'text-decoration': 'none'}),
                    outline=True,
                    style={
                        'textAlign': 'left',
                        'width': '100%',
                    },
                    id={
                        'type': 'navbutton',
                        'module': 'temporal-sequence',
                    }
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
                dbc.NavLink(
                    html.Div([
                        DashIconify(
                            icon="healthicons:medium-bars",
                            color='#f44336',
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Feature Importance"
                    ]),
                    id="feature-importance-link",
                    href="/feature-importance", active="exact"
                ),
                # dimensionality reduction
                dbc.NavLink(
                    html.Div([
                        DashIconify(
                            icon="fluent-mdl2:web-components",
                            color='#f44336',
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Dimensionality Reduction"
                    ]),
                    id="dimensionality-reduction-link",
                    href="/dimensionality-reduction", active="exact"
                ),
                # hyperparameter tuning
                dbc.NavLink(
                    html.Div([
                        DashIconify(
                            icon="zondicons:tuning",
                            color='#f44336',
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Hyperparameter Tuning"
                    ]),
                    id="hyperparameter-tuning-link",
                    href="/hyperparameter-tuning", active="exact"
                ),
                # modeling
                dbc.NavLink(
                    html.Div([
                        DashIconify(
                            icon="fluent-mdl2:server-processes",
                            color='#f44336',
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Modeling"
                    ]),
                    id="modeling-link",
                    href="/modeling", active="exact"
                ),
                # forecasting
                dbc.NavLink(
                    html.Div([
                        DashIconify(
                            icon="carbon:forecast-lightning",
                            color='#f44336',
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Forecasting"
                    ]),
                    id="forecasting-link",
                    href="/forecasting", active="exact"
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
        active=0,
        lineWidth=1,
        color='green',
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
    html.Div(id='current-page', style={'display':'none'}),

], className="dbc")


# @app.callback(
#     Output("page-content", "children"),
#     Input("url", "pathname"),
# )
# def render_page_layout(pathname):
# # DATA
#     if pathname == "/add-data":
#         return add_data
#     elif pathname == "/add-features":
#         return add_features
#     elif pathname == "/preprocessing":
#         return preprocessing

# # ANALYSIS
#     elif pathname == "/price-charts":
#         return price_charts
#     elif pathname == "/distributions":
#         return distributions
#     elif pathname == "/statistical-association":
#         return statistical_association
#     elif pathname == "/temporal-sequence":
#         return temporal_sequence

# # MACHINE LEARNING
#     elif pathname == '/feature-importance':
#         return feature_importance
#     elif pathname == '/dimensionality-reduction':
#         return dimensionality_reduction
#     elif pathname == '/hyperparameter-tuning':
#         return hyperparameter_tuning
#     elif pathname == '/modeling':
#         return modeling
#     elif pathname == '/forecasting':
#         return forecasting


