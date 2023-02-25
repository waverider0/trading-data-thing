import os
import pickle

from dash import dcc, html, ALL
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go

import pandas as pd
import numpy as np
from scipy.stats import norm

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve

from contents.app import *
from contents.libs.cross_validation import cv_score
from contents.storage.utils import delete_temp_models, get_saved_models


@app.callback(
    Output('clf-features', 'options'),
    Input('clf-target', 'value'),
    State('the-data', 'data')
)
def update_clf_features(target, data):
    return [{'label': 'ALL FEATURES', 'value': 'ALL FEATURES'}] + [{'label':i, 'value':i} for i in data[0].keys() if i != target]

@app.callback(
    Output('reg-features', 'options'),
    Input('reg-target', 'value'),
    State('the-data', 'data')
)
def update_reg_features(target, data):
    return [{'label': 'ALL FEATURES', 'value': 'ALL FEATURES'}] + [{'label':i, 'value':i} for i in data[0].keys() if i != target]

@app.callback(
    Output('clf-hyperparam-inputs-container', 'children'),
    Input('clf-estimators', 'value')
)
def render_clf_hyperparam_inputs(estimators):
    children = []
    for estimator in estimators:
        if estimator == 'logreg':
            children.append(html.Div([
                html.Hr(),
                html.H4('Logistic Regression Hyperparameters'),
                html.Div(style={'height': '20px'}),
                html.Div([
                    html.B('C', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Input(
                        id={
                            'estimator': estimator,
                            'param': 'C'
                        },
                        type='number',
                        placeholder='C',
                        value=1.0,
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '20px'}),
                    html.B('Penalty', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Dropdown(
                        id={
                            'estimator': estimator,
                            'param': 'penalty'
                        },
                        options=[
                            {'label': 'l1', 'value': 'l1'},
                            {'label': 'l2', 'value': 'l2'},
                            {'label': 'elasticnet', 'value': 'elasticnet'},
                        ],
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '20px'}),
                    html.B('Solver', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Dropdown(
                        id={
                            'estimator': estimator,
                            'param': 'solver'
                        },
                        options=[
                            'lbfgs',
                            'liblinear',
                            'newton-cg',
                            'newton-cholesky',
                            'sag',
                            'saga'
                        ],
                        style={'width': '100%'}
                    ),
                ], style={'display': 'flex'}),
            ]))
        elif estimator == 'rf':
            children.append(html.Div([
                html.Hr(),
                html.H4('Random Forest Hyperparameters'),
                html.Div(style={'height': '20px'}),
                html.Div([
                    html.B('n_estimators', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Input(
                        id={
                            'estimator': estimator,
                            'param': 'n_estimators'
                        },
                        type='number',
                        placeholder='n_estimators',
                        value=100,
                        step=1,
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '20px'}),
                    html.B('max_depth', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Input(
                        id={
                            'estimator': estimator,
                            'param': 'max_depth'
                        },
                        type='number',
                        placeholder='max_depth',
                        value=3,
                        step=1,
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '20px'}),
                    html.B('min_samples_split', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Input(
                        id={
                            'estimator': estimator,
                            'param': 'min_samples_split'
                        },
                        type='number',
                        placeholder='min_samples_split',
                        value=2,
                        step=1,
                        style={'width': '100%'}
                    ),
                ], style={'display': 'flex'}),
                html.Div(style={'height': '20px'}),
                html.Div([
                    html.B('min_samples_leaf', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Input(
                        id={
                            'estimator': estimator,
                            'param': 'min_samples_leaf'
                        },
                        type='number',
                        placeholder='min_samples_leaf',
                        value=1,
                        step=1,
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '20px'}),
                    html.B('max_features', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Input(
                        id={
                            'estimator': estimator,
                            'param': 'max_features'
                        },
                        type='number',
                        placeholder='max_features',
                        value=1,
                        step=1,
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '20px'}),
                    html.B('bootstrap', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Dropdown(
                        id={
                            'estimator': estimator,
                            'param': 'bootstrap'
                        },
                        options=[
                            {'label': 'True', 'value': True},
                            {'label': 'False', 'value': False},
                        ],
                        style={'width': '100%'}
                    ),
                ], style={'display': 'flex'}),
            ]))
        elif estimator == 'xgb':
            children.append(html.Div([
                html.Hr(),
                html.H4('XGBoost Hyperparameters'),
                html.Div(style={'height': '20px'}),
                html.Div([
                    html.B('n_estimators', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Input(
                        id={
                            'estimator': estimator,
                            'param': 'n_estimators'
                        },
                        type='number',
                        placeholder='n_estimators',
                        value=100,
                        step=1,
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '20px'}),
                    html.B('max_depth', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Input(
                        id={
                            'estimator': estimator,
                            'param': 'max_depth'
                        },
                        type='number',
                        placeholder='max_depth',
                        value=3,
                        step=1,
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '20px'}),
                    html.B('min_child_weight', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Input(
                        id={
                            'estimator': estimator,
                            'param': 'min_child_weight'
                        },
                        type='number',
                        placeholder='min_child_weight',
                        value=1,
                        step=1,
                        style={'width': '100%'}
                    ),
                ], style={'display': 'flex'}),
                html.Div(style={'height': '20px'}),
                html.Div([
                    html.B('gamma', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Input(
                        id={
                            'estimator': estimator,
                            'param': 'gamma'
                        },
                        type='number',
                        placeholder='gamma',
                        value=0,
                        step=0.01,
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '20px'}),
                    html.B('learning_rate', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Input(
                        id={
                            'estimator': estimator,
                            'param': 'learning_rate'
                        },
                        type='number',
                        placeholder='learning_rate',
                        value=0.01,
                        step=0.01,
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '20px'}),
                    html.B('subsample', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Input(
                        id={
                            'estimator': estimator,
                            'param': 'subsample'
                        },
                        type='number',
                        placeholder='subsample',
                        value=0.01,
                        step=0.01,
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '20px'}),
                    html.B('colsample_bytree', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Input(
                        id={
                            'estimator': estimator,
                            'param': 'colsample_bytree'
                        },
                        type='number',
                        placeholder='colsample_bytree',
                        value=0.01,
                        step=0.01,
                        style={'width': '100%'}
                    ),
                ], style={'display': 'flex'}),
            ]))

    return children

@app.callback(
    Output('reg-hyperparam-inputs-container', 'children'),
    Input('reg-estimators', 'value')
)
def render_reg_hyperparam_inputs(estimators):
    children = []
    for estimator in estimators:
        if estimator == 'linreg':
            children.append(html.Div([
                html.Hr(),
                html.H4('Linear Regression Hyperparameters'),
                html.Div(style={'height': '20px'}),
                html.Div([
                    html.B('fit_intercept', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Dropdown(
                        id={
                            'estimator': estimator,
                            'param': 'fit_intercept'
                        },
                        options=[
                            {'label': 'True', 'value': True},
                            {'label': 'False', 'value': False}
                        ],
                    ),
                ]),
            ]))
        elif estimator == 'rf':
            children.append(html.Div([
                html.Hr(),
                html.H4('Random Forest Hyperparameters'),
                html.Div(style={'height': '20px'}),
                html.Div([
                    html.B('n_estimators', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Input(
                        id={
                            'estimator': estimator,
                            'param': 'n_estimators'
                        },
                        type='number',
                        placeholder='n_estimators',
                        value=100,
                        step=1,
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '20px'}),
                    html.B('max_depth', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Input(
                        id={
                            'estimator': estimator,
                            'param': 'max_depth'
                        },
                        type='number',
                        placeholder='max_depth',
                        value=3,
                        step=1,
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '20px'}),
                    html.B('min_samples_split', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Input(
                        id={
                            'estimator': estimator,
                            'param': 'min_samples_split'
                        },
                        type='number',
                        placeholder='min_samples_split',
                        value=2,
                        step=1,
                        style={'width': '100%'}
                    ),
                ], style={'display': 'flex'}),
                html.Div(style={'height': '20px'}),
                html.Div([
                    html.B('min_samples_leaf', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Input(
                        id={
                            'estimator': estimator,
                            'param': 'min_samples_leaf'
                        },
                        type='number',
                        placeholder='min_samples_leaf',
                        value=1,
                        step=1,
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '20px'}),
                    html.B('max_features', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Input(
                        id={
                            'estimator': estimator,
                            'param': 'max_features'
                        },
                        type='number',
                        placeholder='max_features',
                        value=1,
                        step=1,
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '20px'}),
                    html.B('bootstrap', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Dropdown(
                        id={
                            'estimator': estimator,
                            'param': 'bootstrap'
                        },
                        options=[
                            {'label': 'True', 'value': True},
                            {'label': 'False', 'value': False},
                        ],
                        style={'width': '100%'}
                    ),
                ], style={'display': 'flex'}),
            ]))
        elif estimator == 'xgb':
            children.append(html.Div([
                html.Hr(),
                html.H4('XGBoost Hyperparameters'),
                html.Div(style={'height': '20px'}),
                html.Div([
                    html.B('n_estimators', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Input(
                        id={
                            'estimator': estimator,
                            'param': 'n_estimators'
                        },
                        type='number',
                        placeholder='n_estimators',
                        value=100,
                        step=1,
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '20px'}),
                    html.B('max_depth', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Input(
                        id={
                            'estimator': estimator,
                            'param': 'max_depth'
                        },
                        type='number',
                        placeholder='max_depth',
                        value=3,
                        step=1,
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '20px'}),
                    html.B('min_child_weight', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Input(
                        id={
                            'estimator': estimator,
                            'param': 'min_child_weight'
                        },
                        type='number',
                        placeholder='min_child_weight',
                        value=1,
                        step=1,
                        style={'width': '100%'}
                    ),
                ], style={'display': 'flex'}),
                html.Div(style={'height': '20px'}),
                html.Div([
                    html.B('gamma', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Input(
                        id={
                            'estimator': estimator,
                            'param': 'gamma'
                        },
                        type='number',
                        placeholder='gamma',
                        value=0,
                        step=0.01,
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '20px'}),
                    html.B('learning_rate', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Input(
                        id={
                            'estimator': estimator,
                            'param': 'learning_rate'
                        },
                        type='number',
                        placeholder='learning_rate',
                        value=0.01,
                        step=0.01,
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '20px'}),
                    html.B('subsample', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Input(
                        id={
                            'estimator': estimator,
                            'param': 'subsample'
                        },
                        type='number',
                        placeholder='subsample',
                        value=0.01,
                        step=0.01,
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '20px'}),
                    html.B('colsample_bytree', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Input(
                        id={
                            'estimator': estimator,
                            'param': 'colsample_bytree'
                        },
                        type='number',
                        placeholder='colsample_bytree',
                        value=0.01,
                        step=0.01,
                        style={'width': '100%'}
                    ),
                ], style={'display': 'flex'}),
            ]))
        
    return children

#def build_clf():

#def build_reg():

#def update_clf_plots():

#def update_reg_plots():

#def compile_model():

