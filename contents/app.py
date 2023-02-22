import os

from dash import html, dcc
from dash_extensions.enrich import DashProxy, MultiplexerTransform
from dash.dependencies import Input, Output

import dash_table
import dash_bootstrap_components as dbc
import dash_loading_spinners as dls
import dash_mantine_components as dmc
from dash_iconify import DashIconify


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
                dbc.NavLink(
                    html.Div([
                        DashIconify(
                            icon="bi:database-fill-add",
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Add Data"
                    ]),
                    href="/add-data", active="exact"
                ),
                # add features
                dbc.NavLink(
                    html.Div([
                        DashIconify(
                            icon="material-symbols:add-chart-rounded",
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Add Features"
                    ]),
                    href="/add-features", active="exact"
                ),
                # preprocessing
                dbc.NavLink(
                    html.Div([
                        DashIconify(
                            icon="carbon:clean",
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Preprocessing"
                    ]),
                    href="/preprocessing", active="exact"
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
                dbc.NavLink(
                    html.Div([
                        DashIconify(
                            icon="iconoir:candlestick-chart",
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Price Charts"
                    ]),
                    href="/price-charts", active="exact"
                ),
                # distributions
                dbc.NavLink(
                    html.Div([
                        DashIconify(
                            icon="mdi:chart-bell-curve",
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Distributions"
                    ]),
                    href="/distributions", active="exact"
                ),
                # statistical association
                dbc.NavLink(
                    html.Div([
                        DashIconify(
                            icon="fluent:arrow-trending-lines-20-filled",
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Statistical Association"
                    ]),
                    href="/association", active="exact"
                ),
                # temporal sequence
                dbc.NavLink(
                    html.Div([
                        DashIconify(
                            icon="ic:outline-access-time",
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Temporal Sequence"
                    ]),
                    href="/temporal", active="exact"
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
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Feature Importance"
                    ]),
                    href="/feature-importance", active="exact"
                ),
                # dimensionality reduction
                dbc.NavLink(
                    html.Div([
                        DashIconify(
                            icon="fluent-mdl2:web-components",
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Dimensionality Reduction"
                    ]),
                    href="/dimensionality-reduction", active="exact"
                ),
                # hyperparameter tuning
                dbc.NavLink(
                    html.Div([
                        DashIconify(
                            icon="zondicons:tuning",
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Hyperparameter Tuning"
                    ]),
                    href="/hyperparameter-tuning", active="exact"
                ),
                # modeling
                dbc.NavLink(
                    html.Div([
                        DashIconify(
                            icon="fluent-mdl2:server-processes",
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Modeling"
                    ]),
                    href="/modeling", active="exact"
                ),
                # forecasting
                dbc.NavLink(
                    html.Div([
                        DashIconify(
                            icon="carbon:forecast-lightning",
                            width=30,
                            height=30,
                        ),
                        html.Span(style={"margin-left": "10px"}),
                        "Forecasting"
                    ]),
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
    html.Div(id="page-content", style={
        "margin-left": "21rem",
        "padding": "2rem 1rem",
    }),

    # hidden divs (for callbacks that don't return anything)
    html.Div(id='hidden-div', style={'display':'none'}),
    html.Div(id='current-page', style={'display':'none'}),

], className="dbc")


###########
# Layouts #
###########
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def render_page_layout(pathname):
# DATA
    if pathname == "/add-data":
        return add_data
    elif pathname == "/add-features":
        return add_features
    elif pathname == "/preprocessing":
        return preprocessing

# ANALYSIS
    elif pathname == "/price-charts":
        return price_charts
    elif pathname == "/distributions":
        return distributions
    elif pathname == "/statistical-association":
        return statistical_association
    elif pathname == "/temporal-sequence":
        return temporal_sequence

# MACHINE LEARNING
    elif pathname == '/feature-importance':
        return feature_importance
    elif pathname == '/dimensionality-reduction':
        return dimensionality_reduction
    elif pathname == '/hyperparameter-tuning':
        return hyperparameter_tuning
    elif pathname == '/modeling':
        return modeling
    elif pathname == '/forecasting':
        return forecasting



add_data = [
    # header
    html.H1("Add Data"),
    html.Hr(),
    # datatable
    dash_table.DataTable(
        id='add-data-table',
        columns=[{"name": i, "id": i} for i in ['']],
        data=[],
        page_size=20,
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
]
add_features = [
    # header
    html.H1("Add Features"),
    html.Hr(),
    # datatable
    dash_table.DataTable(
        id='add-features-table',
        columns=[{"name": i, "id": i} for i in ['']],
        data=[],
        page_size=5,
        style_table={'overflowX': 'scroll'},
    ),
    # refresh button
    html.Div(style={'height': '10px'}),
    dbc.Button(
        "Refresh",
        id="add-features-refresh",
        color="primary",
        className="mr-1",
        style={'width': '100%'}
    ),
    html.Div(style={'height': '10px'}),
    # accordion
    dbc.Accordion([
        dbc.AccordionItem(
            title="Type Casting",
        ),
        dbc.AccordionItem(
            title="Drop Features",
        ),
        dbc.AccordionItem(
            title="Transformations",
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
    ], start_collapsed=True)
]
preprocessing = [
    # header
    html.H1("Preprocessing"),
    html.Hr(),
    # datatable
    dash_table.DataTable(
        id='preprocessing-table',
        columns=[{"name": i, "id": i} for i in ['']],
        data=[],
        page_size=5,
        style_table={'overflowX': 'scroll'},
    ),
    # refresh button
    html.Div(style={'height': '10px'}),
    dbc.Button(
        "Refresh",
        id="preprocessing-refresh",
        color="primary",
        className="mr-1",
        style={'width': '100%'}
    ),
    html.Div(style={'height': '10px'}),
    # accordion
    dbc.Accordion([
        dbc.AccordionItem(
            title="Missing Values",
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
    # header
    html.H1("Price Charts"),
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
distributions = []
statistical_association = []
temporal_sequence = []

feature_importance = []
dimensionality_reduction = []
hyperparameter_tuning = []
modeling = []
forecasting = []