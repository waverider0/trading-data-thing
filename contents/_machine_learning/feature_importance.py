import math
import pandas as pd
import numpy as np

from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import dash_daq as daq
import plotly.express as px
import plotly.graph_objs as go

from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import (
    make_scorer,
    # classification
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    # regression
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder

from contents.app import *
from contents.libs.boruta_shap import BorutaShap

@app.callback(
    Output('rfe-inputs-container', 'children'),
    Input('rfe-model-type', 'value'),
    State('the-data', 'data')
)
def render_rfe_inputs(model_type, data):
    if model_type and data:
        df = pd.DataFrame.from_dict(data)
        if model_type == 'clf':
            return [
                html.Div([
                    html.B('Target', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Dropdown(
                        id='rfe-target',
                        options=[{'label': col, 'value': col} for col in df.columns],
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '20px'}),
                    html.B('Estimator', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Dropdown(
                        id='rfe-estimator',
                        options=[
                            {'label': 'Logistic Regression', 'value': 'lr'},
                            {'label': 'Random Forest', 'value': 'rf'},
                            {'label': 'XGBoost', 'value': 'xgb'},
                        ],
                        style={'width': '100%'}
                    ),
                ], style={'display': 'flex'}),
                html.Div(style={'height': '10px'}),
                html.Div([
                    html.B('Scoring', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Dropdown(
                        id='rfe-scoring',
                        options=[
                            {'label': 'Accuracy', 'value': 'accuracy'},
                            {'label': 'Precision', 'value': 'precision'},
                            {'label': 'Recall', 'value': 'recall'},
                            {'label': 'F1', 'value': 'f1'},
                        ],
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '20px'}),
                    html.B('CV', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Dropdown(
                        id='rfe-cv',
                        options=[
                            {'label': 'KFold', 'value': 'kfold'},
                            {'label': 'Stratified KFold', 'value': 'skfold'},
                            {'label': 'Time Series Split', 'value': 'tss'},
                            {'label': 'CPCV', 'value': 'cpcv'},
                        ],
                        style={'width': '100%'}
                    ),
                ], style={'display': 'flex'}),
                html.Div(style={'height': '10px'}),
                dbc.Button(
                    'Run RFE',
                    id='rfe-run',
                    color='primary',
                    style={'width': '100%'}
                )
            ]
        return [
            html.Div([
                html.B('Target', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                html.Div(style={'width': '10px'}),
                dcc.Dropdown(
                    id='rfe-target',
                    options=[{'label': col, 'value': col} for col in df.columns],
                    style={'width': '100%'}
                ),
                html.Div(style={'width': '20px'}),
                html.B('Estimator', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                html.Div(style={'width': '10px'}),
                dcc.Dropdown(
                    id='rfe-estimator',
                    options=[
                        {'label': 'Linear Regression', 'value': 'lr'},
                        {'label': 'Random Forest', 'value': 'rf'},
                        {'label': 'XGBoost', 'value': 'xgb'},
                    ],
                    style={'width': '100%'}
                ),
            ], style={'display': 'flex'}),
            html.Div(style={'height': '10px'}),
            html.Div([
                html.B('Scoring', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                html.Div(style={'width': '10px'}),
                dcc.Dropdown(
                    id='rfe-scoring',
                    options=[
                        {'label': 'MSE', 'value': 'mse'},
                        {'label': 'RMSE', 'value': 'rmse'},
                        {'label': 'MAE', 'value': 'mae'},
                        {'label': 'R2', 'value': 'r2'},
                    ],
                    style={'width': '100%'}
                ),
                html.Div(style={'width': '20px'}),
                html.B('CV', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                html.Div(style={'width': '10px'}),
                dcc.Dropdown(
                    id='rfe-cv',
                    options=[
                        {'label': 'KFold', 'value': 'kfold'},
                        {'label': 'Stratified KFold', 'value': 'skfold'},
                        {'label': 'Time Series Split', 'value': 'tss'},
                        {'label': 'CPCV', 'value': 'cpcv'},
                    ],
                    style={'width': '100%'}
                ),
            ], style={'display': 'flex'}),
            html.Div(style={'height': '10px'}),
            dbc.Button(
                'Run RFE',
                id='rfe-run',
                color='primary',
                style={'width': '100%'}
            )
        ]
    raise PreventUpdate