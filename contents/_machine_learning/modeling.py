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
        if estimator == 'lr_clf':
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
        elif estimator == 'rf_clf':
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
        elif estimator == 'xgb_clf':
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
        if estimator == 'lr_reg':
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
        elif estimator == 'rf_reg':
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
        elif estimator == 'xgb_reg':
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
    Output('clf-outputs-container', 'children'),
    Output('clf-results', 'data'),
    Input('build-clf-model', 'n_clicks'),
    State('clf-target', 'value'),
    State('clf-features', 'value'),
    State('clf-estimators', 'value'),
    State('clf-cv', 'value'),
    State('calibration-method', 'value'),
    State({'estimator': 'lr_clf', 'param': ALL}, 'value'),
    State({'estimator': 'rf_clf', 'param': ALL}, 'value'),
    State({'estimator': 'xgb_clf', 'param': ALL}, 'value'),
    State('the-data', 'data')
)
def build_clf(
    n_clicks,
    target,
    features,
    estimators,
    cv,
    calibration_method,
    lr_params,
    rf_params,
    xgb_params,
    data
):
    if (n_clicks and
        target and
        features and
        estimators and
        cv and
        calibration_method and
        data
    ):
        if features == ['ALL FEATURES']: features = [col for col in data[0].keys() if col != target]
        elif len(features) > 1 and 'ALL FEATURES' in features: features.remove('ALL FEATURES')
        df = pd.DataFrame.from_dict(data)
        X = df[features]
        y = pd.Series(LabelEncoder().fit_transform(df[target]))

        # Populate hyperparameter dictionaries
        lr_param_grid = {}
        rf_param_grid = {}
        xgb_param_grid = {}
        if len(lr_params) > 0:
            lr_param_grid = {
                'C': lr_params[0],
                'penalty': lr_params[1],
                'solver': lr_params[2],
            }
        if len(rf_params) > 0:
            rf_param_grid = {
                'n_estimators': rf_params[0],
                'max_depth': rf_params[1],
                'min_samples_split': rf_params[2],
                'min_samples_leaf': rf_params[3],
                'max_features': rf_params[4],
                'bootstrap': rf_params[5],

            }
        if len(xgb_params) > 0:
            xgb_param_grid = {
                'n_estimators': xgb_params[0],
                'max_depth': xgb_params[1],
                'min_child_weight': xgb_params[2],
                'gamma': xgb_params[3],
                'learning_rate': xgb_params[4],
                'subsample': xgb_params[5],
                'colsample_bytree': xgb_params[6],
            }

        estimator_mappings = {
            'lr_clf': LogisticRegression(**lr_param_grid),
            'rf_clf': RandomForestClassifier(**rf_param_grid),
            'xgb_clf': XGBClassifier(**xgb_param_grid)
        }
        model = None
        if len(estimators) > 1: # ensemble model
            model = StackingClassifier(
                final_estimator=RandomForestClassifier(),
                estimators=[(estimator, estimator_mappings[estimator]) for estimator in estimators]
            )
            if calibration_method != 'none':
                if cv == 'kfold':
                    model = CalibratedClassifierCV(
                        model,
                        method=calibration_method,
                        cv=KFold(n_splits=5, shuffle=True, random_state=69)
                    )
                elif cv == 'skfold':
                    model = CalibratedClassifierCV(
                        model,
                        method=calibration_method,
                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=69)
                    )
                elif cv == 'tss' or cv == 'cpcv':
                    model = CalibratedClassifierCV(
                        model,
                        method=calibration_method,
                        cv=TimeSeriesSplit(n_splits=5)
                    )
        else: # single model
            model = estimator_mappings[estimators[0]]
            if calibration_method != 'none':
                if cv == 'kfold':
                    model = CalibratedClassifierCV(
                        model,
                        method=calibration_method,
                        cv=KFold(n_splits=5, shuffle=True, random_state=69)
                    )
                elif cv == 'skfold':
                    model = CalibratedClassifierCV(
                        model,
                        method=calibration_method,
                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=69)
                    )
                elif cv == 'tss' or cv == 'cpcv':
                    model = CalibratedClassifierCV(
                        model,
                        method=calibration_method,
                        cv=TimeSeriesSplit(n_splits=5)
                    )
    
        modeling_results = cv_score(
            X,
            y,
            model,
            None,
            method=cv,
            return_raw=True,
        )

        save_temp_model(
            model.fit(df[features], df[target]),
            features,
            'temp_clf'
        )

        return [
            html.Div([
                html.H4('Classification Model Interpretability'),
                dcc.Dropdown(
                    id='clf-plots-dropdown',
                    options=[
                        {'label': 'Confusion Matrix', 'value': 'confusion_matrix'},
                        {'label': 'ROC Curve', 'value': 'roc_curve'},
                        {'label': 'Precision-Recall Curve', 'value': 'pr_curve'},
                        {'label': 'Calibration Curve', 'value': 'calibration_curve'},
                    ]
                ),
                html.Br(),
                html.Div(id='clf-plots-container'),
                html.Hr(),
                html.Div([
                    html.B('Model Name', style={'margin-top': '3px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Input(
                        id='clf-model-name',
                        placeholder='Enter a name for your model...',
                        type='text',
                        value='',
                        style={'width': '100%'}
                    )
                ], style={'display': 'flex'}),
                html.Div(style={'height': '20px'}),
                dbc.Button(
                    'Compile Model',
                    id='compile-clf-model-button',
                    color='primary',
                    style={'width': '100%'}
                )
            ]),
            modeling_results,
        ]

    raise PreventUpdate

@app.callback(
    Output('reg-outputs-container', 'children'),
    Output('reg-results', 'data'),
    Input('build-reg-model', 'n_clicks'),
    State('reg-target', 'value'),
    State('reg-features', 'value'),
    State('reg-estimators', 'value'),
    State('reg-cv', 'value'),
    State({'estimator': 'lr_reg', 'param': ALL}, 'value'),
    State({'estimator': 'rf_reg', 'param': ALL}, 'value'),
    State({'estimator': 'xgb_reg', 'param': ALL}, 'value'),
    State('the-data', 'data')
)
def build_reg(
    n_clicks,
    target,
    features,
    estimators,
    cv,
    lr_params,
    rf_params,
    xgb_params,
    data
):
    if (n_clicks and
        target and
        features and
        estimators and
        cv and
        data
    ):
        if features == ['ALL FEATURES']: features = [col for col in data[0].keys() if col != target]
        elif len(features) > 1 and 'ALL FEATURES' in features: features.remove('ALL FEATURES')
        df = pd.DataFrame.from_dict(data)
        X = df[features]
        y = df[target]

        # Populate hyperparameter dictionaries
        lr_param_grid = {}
        rf_param_grid = {}
        xgb_param_grid = {}
        if len(lr_params) > 0:
            lr_param_grid = {
                'fit_intercept': lr_params[0]
            }
        if len(rf_params) > 0:
            rf_param_grid = {
                'n_estimators': rf_params[0],
                'max_depth': rf_params[1],
                'min_samples_split': rf_params[2],
                'min_samples_leaf': rf_params[3],
                'max_features': rf_params[4],
                'bootstrap': rf_params[5],

            }
        if len(xgb_params) > 0:
            xgb_param_grid = {
                'n_estimators': xgb_params[0],
                'max_depth': xgb_params[1],
                'min_child_weight': xgb_params[2],
                'gamma': xgb_params[3],
                'learning_rate': xgb_params[4],
                'subsample': xgb_params[5],
                'colsample_bytree': xgb_params[6],
            }
        
        estimator_mappings = {
            'lr_reg': LinearRegression(**lr_param_grid),
            'rf_reg': RandomForestRegressor(**rf_param_grid),
            'xgb_reg': XGBRegressor(**xgb_param_grid)
        }
        model = None
        if len(estimators) > 1: # ensemble model
            model = StackingRegressor(
                final_estimator=RandomForestRegressor(),
                estimators=[(estimator, estimator_mappings[estimator]) for estimator in estimators]
            )
        else: # single model
            model = estimator_mappings[estimators[0]]

        modeling_results = cv_score(
            X,
            y,
            model,
            None,
            method=cv,
            return_raw=True,
        )

        save_temp_model(
            model.fit(df[features], df[target]),
            features,
            'temp_reg'
        )

        return [
            html.Div([
                html.H4('Regression Model Interpretability'),
                dcc.Dropdown(
                    id='reg-plots-dropdown',
                    options=[
                        {'label': 'Prediction vs. Actual', 'value': 'pred_vs_actual'},
                        {'label': 'Residuals', 'value': 'residuals'},
                        {'label': 'Residual Distribution', 'value': 'residual_dist'},
                    ]
                ),
                html.Br(),
                html.Div(id='reg-plots-container'),
                html.Hr(),
                html.Div([
                    html.B('Model Name', style={'margin-top': '3px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Input(
                        id='reg-model-name',
                        placeholder='Enter a name for your model...',
                        type='text',
                        value='',
                        style={'width': '100%'}
                    )
                ], style={'display': 'flex'}),
                html.Div(style={'height': '20px'}),
                dbc.Button(
                    'Compile Model',
                    id='compile-reg-model-button',
                    color='primary',
                    style={'width': '100%'}
                )
            ]),
            modeling_results,
        ]

    raise PreventUpdate

@app.callback(
    Output('clf-plots-container', 'children'),
    Input('clf-plots-dropdown', 'value'),
    State('clf-results', 'data'),
)
def update_clf_plots(plot_type, modeling_results):
    if plot_type and modeling_results:
        if len(set(modeling_results['true_vals'])) == 2:
            if plot_type == 'confusion_matrix':
                conf_matrix = pd.crosstab(
                    modeling_results['true_vals'],
                    modeling_results['preds'],
                    rownames=['Actual'],
                    colnames=['Predicted']
                )
                return dcc.Graph(
                    figure=go.Figure(
                        data=go.Heatmap(
                            z=conf_matrix.values,
                            x=conf_matrix.columns,
                            y=conf_matrix.index,
                            colorscale='thermal',
                        ),
                        layout=go.Layout(
                            title='Confusion Matrix',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#FFFFFF'),
                            xaxis=dict(
                                title='Predicted',
                                titlefont=dict(color='#FFFFFF'),
                                showgrid=False,
                            ),
                            yaxis=dict(
                                title='Actual',
                                titlefont=dict(color='#FFFFFF'),
                                showgrid=False,
                            ),
                        )
                    )
                )

            elif plot_type == 'roc_curve':
                fpr, tpr, thresholds = roc_curve(modeling_results['true_vals'], modeling_results['pred_probas'])
                auc_value = auc(fpr, tpr)
                # Empirical curve
                graph = dcc.Graph(
                    figure=go.Figure(
                        data=go.Scatter(
                            x=fpr,
                            y=tpr,
                            mode='lines',
                            name=f'Empirical Curve (AUC = {auc_value:.3f})',
                            line=dict(color='#37699b'),
                        ),
                        layout=go.Layout(
                            title='ROC Curve',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#FFFFFF'),
                            xaxis=dict(
                                title='False Positive Rate',
                                titlefont=dict(color='#FFFFFF'),
                                showgrid=False,
                            ),
                            yaxis=dict(
                                title='True Positive Rate',
                                titlefont=dict(color='#FFFFFF'),
                                showgrid=False,
                            ),
                        )
                    )
                )
                # Perfect curve
                graph.figure.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[1, 1],
                        mode='lines',
                        name='Perfect Curve (AUC = 1.000)',
                        line=dict(
                            color='orange',
                            dash='dash',
                        )
                    )
                )
                graph.figure.add_shape(
                    type='line',
                    x0=0,
                    y0=0,
                    x1=0,
                    y1=1,
                    line=dict(
                        color='orange',
                        dash='dash',
                    )
                )
                graph.figure.update_layout(showlegend=True)
                return graph

            elif plot_type == 'pr_curve':
                precision, recall, thresholds = precision_recall_curve(modeling_results['true_vals'], modeling_results['pred_probas'])
                # Empirical curve
                graph = dcc.Graph(
                    figure=go.Figure(
                        data=go.Scatter(
                            x=recall,
                            y=precision,
                            mode='lines',
                            name='Empirical Curve',
                            line=dict(color='#37699b'),
                        ),
                        layout=go.Layout(
                            title='Precision-Recall Curve',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#FFFFFF'),
                            xaxis=dict(
                                title='Recall',
                                titlefont=dict(color='#FFFFFF'),
                                showgrid=False,
                            ),
                            yaxis=dict(
                                title='Precision',
                                titlefont=dict(color='#FFFFFF'),
                                showgrid=False,
                            ),
                        )
                    )
                )
                # Perfect curve
                graph.figure.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[1, 1],
                        mode='lines',
                        name='Perfect Curve',
                        line=dict(
                            color='orange',
                            dash='dash',
                        )
                    )
                )
                graph.figure.add_shape(
                    type='line',
                    x0=1,
                    y0=0,
                    x1=1,
                    y1=1,
                    line=dict(
                        color='orange',
                        dash='dash',
                    )
                )
                graph.figure.update_layout(showlegend=True)
                return graph

            elif plot_type == 'calibration_curve':
                prob_true, prob_pred = calibration_curve(modeling_results['true_vals'], modeling_results['pred_probas'], n_bins=10)
                # Empirical curve
                graph = dcc.Graph(
                    figure=go.Figure(
                        data=go.Scatter(
                            x=prob_pred,
                            y=prob_true,
                            mode='lines',
                            name='Empirical Curve',
                            line=dict(color='#37699b'),
                        ),
                        layout=go.Layout(
                            title='Calibration Curve',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#FFFFFF'),
                            xaxis=dict(
                                title='Predicted Probability',
                                titlefont=dict(color='#FFFFFF'),
                                showgrid=False,
                            ),
                            yaxis=dict(
                                title='True Probability',
                                titlefont=dict(color='#FFFFFF'),
                                showgrid=False,
                            ),
                        )
                    )
                )
                # Perfect curve
                graph.figure.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode='lines',
                        name='Perfect Curve',
                        line=dict(
                            color='orange',
                            dash='dash',
                        )
                    )
                )
                graph.figure.update_layout(showlegend=True)
                return graph

        return html.Div('No plots available for multiclass classification')
    
    raise PreventUpdate

@app.callback(
    Output('reg-plots-container', 'children'),
    Input('reg-plots-dropdown', 'value'),
    State('reg-results', 'data'),
)
def update_reg_plots(plot_type, modeling_results):
    if plot_type and modeling_results:
        if plot_type == 'pred_vs_actual':
            # Empirical
            graph = dcc.Graph(
                figure=go.Figure(
                    data=go.Scatter(
                        x=modeling_results['preds'],
                        y=modeling_results['true_vals'],
                        mode='markers',
                        name='Empirical',
                        marker=dict(color='#37699b'),
                    ),
                    layout=go.Layout(
                        title='Prediction vs. Actual',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#FFFFFF'),
                        xaxis=dict(
                            title='Predicted Value',
                            titlefont=dict(color='#FFFFFF'),
                            showgrid=False,
                        ),
                        yaxis=dict(
                            title='Actual Value',
                            titlefont=dict(color='#FFFFFF'),
                            showgrid=False,
                        ),
                    )
                )
            )
            # Perfect
            graph.figure.add_trace(
                go.Scatter(
                    x=[min(modeling_results['true_vals']), max(modeling_results['true_vals'])],
                    y=[min(modeling_results['true_vals']), max(modeling_results['true_vals'])],
                    mode='lines',
                    name='Perfect Fit',
                    line=dict(
                        color='orange',
                        dash='dash',
                    )
                )
            )
            graph.figure.update_layout(showlegend=True)
            return graph

        elif plot_type == 'residuals':
            # Empirical
            graph = dcc.Graph(
                figure=go.Figure(
                    data=go.Scatter(
                        x=modeling_results['residuals'],
                        y=modeling_results['preds'],
                        mode='markers',
                        name='Empirical',
                        marker=dict(color='#37699b'),
                    ),
                    layout=go.Layout(
                        title='Residuals Plot',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#FFFFFF'),
                        xaxis=dict(
                            title='Error',
                            titlefont=dict(color='#FFFFFF'),
                            showgrid=False,
                        ),
                        yaxis=dict(
                            title='Predicted Value',
                            titlefont=dict(color='#FFFFFF'),
                            showgrid=False,
                        ),
                        xaxis_range=[
                            min([x * -1 if x > 0 else x for x in modeling_results['preds']]),
                            max([x * -1 if x < 0 else x for x in modeling_results['preds']])
                        ]
                    )
                )
            )
            # Perfect
            graph.figure.add_trace(
                go.Scatter(
                    x=[0, 0],
                    y=[min(modeling_results['preds']), max(modeling_results['preds'])],
                    mode='lines',
                    name='Perfect Fit',
                    line=dict(
                        color='orange',
                        dash='dash',
                    )
                )
            )
            graph.figure.update_layout(showlegend=True)
            return graph

        elif plot_type == 'residual_dist':
            residuals = modeling_results['residuals']
            # Perfect Normal
            mu, std = norm.fit(residuals)
            x = np.linspace(min(residuals), max(residuals), 100)
            p = norm.pdf(x, mu, std)
            graph = dcc.Graph(
                figure=go.Figure(
                    data=go.Scatter(
                        x=x,
                        y=p,
                        mode='lines',
                        name='Normal Distribution',
                        line=dict(color='orange'),
                    ),
                    layout=go.Layout(
                        title='Residual vs Normal Distribution',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#FFFFFF'),
                        xaxis=dict(
                            title='Error',
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
            # Empirical
            graph.figure.add_trace(
                go.Histogram(
                    x=residuals,
                    name='Empirical',
                    histnorm='probability density',
                    marker=dict(color='#37699b'),
                )
            )
            graph.figure.update_layout(showlegend=True)
            return graph

    raise PreventUpdate

@app.callback(
    Output('pre-compiled-models', 'options'),
    Input('compile-clf-model-button', 'n_clicks'),
    State('clf-model-name', 'value'),
)
def compile_classifier(n_clicks, model_name):
    if n_clicks:
        dirname = os.path.dirname(os.path.dirname(__file__))
        os.rename(
            f'{dirname}/storage/temp/temp_clf',
            f'{dirname}/storage/saved_models/{model_name}'
        )
        return get_saved_models()
    raise PreventUpdate

@app.callback(
    Output('pre-compiled-models', 'options'),
    Input('compile-reg-model-button', 'n_clicks'),
    State('reg-model-name', 'value'),
)
def compile_regressor(n_clicks, model_name):
    if n_clicks:
        dirname = os.path.dirname(os.path.dirname(__file__))
        os.rename(
            f'{dirname}/storage/temp/temp_reg',
            f'{dirname}/storage/saved_models/{model_name}'
        )
        return get_saved_models()
    raise PreventUpdate

@app.callback(
    Output('clf-outputs-container', 'children'),
    Input('clf-target', 'value'),
    Input('clf-features', 'value'),
    Input('clf-estimators', 'value'),
    Input('clf-cv', 'value'),
    Input('calibration-method', 'value'),
)
def clear_clf_outputs_container(target, features, estimators, cv, calibration_method):
    if target or features or estimators or cv or calibration_method:
        return ''
    raise PreventUpdate

@app.callback(
    Output('reg-outputs-container', 'children'),
    Input('reg-target', 'value'),
    Input('reg-features', 'value'),
    Input('reg-estimators', 'value'),
    Input('reg-cv', 'value'),
)
def clear_reg_outputs_container(target, features, estimators, cv):
    if target or features or estimators or cv:
        return ''
    raise PreventUpdate


# Helper Methods
def save_temp_model(model, features, model_name):
    """ Saves the model and features used to train the model to a temp folder
    """
    delete_temp_models(model_name)
    dirname = os.path.dirname(os.path.dirname(__file__))
    features = [str(feature) for feature in features]

    os.mkdir(f'{dirname}/storage/temp/{model_name}')

    with open(f'{dirname}/storage/temp/{model_name}/compiled_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open(f'{dirname}/storage/temp/{model_name}/features.txt', 'w') as f:
        f.write(','.join(features))

