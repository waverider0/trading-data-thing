import math
from functools import partial
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.express as px
import plotly.graph_objs as go

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    make_scorer,
    # Classification
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    # Regression
    mean_squared_error,
    mean_absolute_error
)

from contents.app import *
from contents.libs.hypertune import RandomSearch, OptunaSearch


@app.callback(
    Output("search-results-plot", "figure"),
    Input('hyperparameters', 'value'),
    State('hyperparameter-search-results', 'data'),
)
def update_hyperparameter_plot(hyperparam, search_results):
    df = pd.DataFrame.from_dict(search_results)
    return go.Figure(
        data=[
            go.Scatter(
                x=df[hyperparam],
                y=df['train_score'],
                mode='markers',
                name='Train',
                marker=dict(color='#37699b')
            ),
            go.Scatter(
                x=df[hyperparam],
                y=df['test_score'],
                mode='markers',
                name='Test'
            )
        ],
        layout=go.Layout(
            title='Hyperparameter Search Results',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF'),
            xaxis=dict(
                title=hyperparam,
                titlefont=dict(color='#FFFFFF'),
                showgrid=False,
            ),
            yaxis=dict(
                title='Score',
                titlefont=dict(color='#FFFFFF'),
                showgrid=False,
            ),
        ),
    )


# Random Search
@app.callback(
    Output("rand-search-inputs-container", "children"),
    Input('rand-search-model-type', 'value'),
    State('the-data', 'data'),
)
def render_rand_search_inputs(model_type, data):
    if model_type and data:
        df = pd.DataFrame.from_dict(data)
        features = [{'label': col, 'value': col} for col in df.columns]
        if model_type == 'clf':
            return [
                html.Div([
                    html.B('Target', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Dropdown(
                        id='rand-search-target',
                        options=features,
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '10px'}),
                    html.B('Features', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Dropdown(
                        id='rand-search-features',
                        options=[{'label': 'ALL FEATURES', 'value': 'ALL FEATURES'}] + features,
                        multi=True,
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '20px'}),
                    html.B('Number of iterations', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(dcc.Slider(
                        id='rand-search-n-iterations',
                        min=1,
                        max=100,
                        step=1,
                        value=10,
                        marks=None,
                        tooltip={'always_visible': False, 'placement': 'bottom'},
                    ), style={'width': '100%', 'margin-top': '12px'}),
                ], style={'display': 'flex'}),
                html.Div(style={'height': '5px'}),
                html.Div([
                    html.B('Estimator', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Dropdown(
                        id='rand-search-estimator',
                        options=[
                            {'label': 'Logistic Regression', 'value': 'logreg'},
                            {'label': 'Random Forest', 'value': 'rf'},
                            {'label': 'XGBoost', 'value': 'xgb'},
                        ],
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '20px'}),
                    html.B('Scoring', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Dropdown(
                        id='rand-search-scoring',
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
                        id='rand-search-cv',
                        options=[
                            {'label': 'KFold', 'value': 'kfold'},
                            {'label': 'StratifiedKFold', 'value': 'skfold'},
                            {'label': 'TimeSeriesSplit', 'value': 'tssplit'},
                            {'label': 'CPCV', 'value': 'cpcv'},
                        ],
                        style={'width': '100%'}
                    ),
                ], style={'display': 'flex'}),
                html.Div(style={'height': '10px'}),
                dbc.Button(
                    "Run Search",
                    id="rand-search-run",
                    color="primary",
                    style={'width': '100%'}
                )
            ]
        return [
            html.Div([
                html.B('Target', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                html.Div(style={'width': '10px'}),
                dcc.Dropdown(
                    id='rand-search-target',
                    options=features,
                    style={'width': '100%'}
                ),
                html.Div(style={'width': '10px'}),
                html.B('Features', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                html.Div(style={'width': '10px'}),
                dcc.Dropdown(
                    id='rand-search-features',
                    options=[{'label': 'ALL FEATURES', 'value': 'ALL FEATURES'}] + features,
                    multi=True,
                    style={'width': '100%'}
                ),
                html.Div(style={'width': '20px'}),
                html.B('Number of iterations', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                html.Div(dcc.Slider(
                    id='rand-search-n-iterations',
                    min=1,
                    max=100,
                    step=1,
                    value=10,
                    marks=None,
                    tooltip={'always_visible': False, 'placement': 'bottom'},
                ), style={'width': '100%', 'margin-top': '12px'}),
            ], style={'display': 'flex'}),
            html.Div(style={'height': '5px'}),
            html.Div([
                html.B('Estimator', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                html.Div(style={'width': '10px'}),
                dcc.Dropdown(
                    id='rand-search-estimator',
                    options=[
                        {'label': 'Linear Regression', 'value': 'linreg'},
                        {'label': 'Random Forest', 'value': 'rf'},
                        {'label': 'XGBoost', 'value': 'xgb'},
                    ],
                    style={'width': '100%'}
                ),
                html.Div(style={'width': '20px'}),
                html.B('Scoring', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                html.Div(style={'width': '10px'}),
                dcc.Dropdown(
                    id='rand-search-scoring',
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
                    id='rand-search-cv',
                    options=[
                        {'label': 'KFold', 'value': 'kfold'},
                        {'label': 'TimeSeriesSplit', 'value': 'tssplit'},
                        {'label': 'CPCV', 'value': 'cpcv'},
                    ],
                    style={'width': '100%'}
                ),
            ], style={'display': 'flex'}),
            html.Div(style={'height': '10px'}),
            dbc.Button(
                "Run Search",
                id="rand-search-run",
                color="primary",
                style={'width': '100%'}
            )
        ]
    raise PreventUpdate

@app.callback(
    Output('rand-search-features', 'options'),
    Input('rand-search-target', 'value'),
    State('the-data', 'data')
)
def update_rand_search_features(target, data):
    return [{'label': 'ALL FEATURES', 'value': 'ALL FEATURES'}] + [{'label':i, 'value':i} for i in data[0].keys() if i != target]

@app.callback(
    Output('rand-search-plot-container', 'children'),
    Output('hyperparameter-search-results', 'data'),
    Input('rand-search-run', 'n_clicks'),
    State('rand-search-model-type', 'value'),
    State('rand-search-target', 'value'),
    State('rand-search-features', 'value'),
    State('rand-search-n-iterations', 'value'),
    State('rand-search-estimator', 'value'),
    State('rand-search-scoring', 'value'),
    State('rand-search-cv', 'value'),
    State('the-data', 'data')
)
def run_rand_search(n_clicks, model_type, target, features, n_iter, estimator, scoring, cv, data):
    if n_clicks and model_type and target and features and n_iter and estimator and scoring and cv and data:
        if features == ['ALL FEATURES']: features = [col for col in data[0].keys() if col != target]
        elif len(features) > 1 and 'ALL FEATURES' in features: features.remove('ALL FEATURES')
        df = pd.DataFrame.from_dict(data)
        df = df.iloc[:int(len(df) * 0.75)]
        X = df[features]
        y = df[target]
        n_features = len(features)

        param_grid_mappings = {
            'linreg': {
                'fit_intercept': [True, False],
            },
            'logreg': {
                'C': np.linspace(0.01, 10e6, 1000, dtype=float),
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': [
                    'lbfgs',
                    'liblinear',
                    'newton-cg',
                    'newton-cholesky',
                    'sag',
                    'saga',
                ],
            },
            'rf': {
                'n_estimators': np.linspace(50, 1000, 100, dtype=int),
                'max_depth': np.linspace(1, 30, 30, dtype=int),
                'min_samples_split': np.linspace(2, 20, 20, dtype=int),
                'min_samples_leaf': np.linspace(1, 20, 20, dtype=int),
                'max_features': np.linspace(math.floor(np.sqrt(n_features)), n_features, n_features, dtype=int),
                'bootstrap': [True, False],
            },
            'xgb': {
                'n_estimators': np.linspace(50, 1000, 100, dtype=int),
                'max_depth': np.linspace(1, 30, 30, dtype=int),
                'min_child_weight': np.linspace(1, 20, 20, dtype=int),
                'gamma': np.linspace(0, 1, 25, dtype=float),
                'learning_rate': np.linspace(0.01, 1, 25, dtype=float),
                'subsample': np.linspace(0.01, 1, 25, dtype=float),
                'colsample_bytree': np.linspace(0.01, 1, 25, dtype=float),
            }
        }

        greater_is_better = True
        scorer = None
        if model_type == 'clf':
            classifiers = {
                'logreg': LogisticRegression(),
                'rf': RandomForestClassifier(),
                'xgb': XGBClassifier(),
            }
            model = classifiers[estimator]
            param_grid = param_grid_mappings[estimator]
            y = pd.Series(LabelEncoder().fit_transform(y))
            if isBinary(y): # binary classification
                scorer_mappings = {
                    'accuracy': make_scorer(accuracy_score),
                    'precision': make_scorer(precision_score),
                    'recall': make_scorer(recall_score),
                    'f1': make_scorer(f1_score),
                }
                scorer = scorer_mappings[scoring]
            else: # multiclass classification
                scorer_mappings = {
                    'accuracy': make_scorer(accuracy_score),
                    'precision': make_scorer(partial(precision_score, average='weighted')),
                    'recall': make_scorer(partial(recall_score, average='weighted')),
                    'f1': make_scorer(partial(f1_score, average='weighted')),
                }
                scorer = scorer_mappings[scoring]
        else: # regression
            regressors = {
                'linreg': LinearRegression(),
                'rf': RandomForestRegressor(),
                'xgb': XGBRegressor(),
            }
            model = regressors[estimator]
            param_grid = param_grid_mappings[estimator]
            greater_is_better = False
            scorer_mappings = {
                'mse': make_scorer(mean_squared_error, greater_is_better=greater_is_better),
                'rmse': make_scorer(partial(mean_squared_error, squared=False), greater_is_better=greater_is_better),
                'mae': make_scorer(mean_absolute_error, greater_is_better=greater_is_better),
            }
            scorer = scorer_mappings[scoring]

        results = RandomSearch(model, cv, scorer, greater_is_better, param_grid, n_iter=n_iter).tune(X, y)
        params = [col for col in results.columns if col not in ['train_score', 'test_score']]
        return [
            [
                html.B('Select Hyperparameter', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                html.Div(style={'width': '10px'}),
                dcc.Dropdown(
                    id='hyperparameters',
                    options=[param for param in params],
                ),
                html.Div(style={'height': '20px'}),
                dcc.Graph(
                    id='search-results-plot',
                    figure=go.Figure(
                        data=[go.Scatter(x=[], y=[])],
                        layout=go.Layout(
                            title='Hyperparameter Search Results',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#FFFFFF'),
                            xaxis=dict(
                                titlefont=dict(color='#FFFFFF'),
                                showgrid=False,
                            ),
                            yaxis=dict(
                                titlefont=dict(color='#FFFFFF'),
                                showgrid=False,
                            ),
                        )
                    )
                )
            ],
            results.to_dict('records'),
        ]
    
    raise PreventUpdate

# Optuna
@app.callback(
    Output('optuna-inputs-container', 'children'),
    Input('optuna-model-type', 'value'),
    State('the-data', 'data')
)
def update_optuna_inputs(model_type, data):
    if model_type and data:
        df = pd.DataFrame.from_dict(data)
        features = [{'label': col, 'value': col} for col in df.columns]
        if model_type == 'clf':
            return [
                html.Div([
                    html.B('Target', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Dropdown(
                        id='optuna-target',
                        options=features,
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '10px'}),
                    html.B('Features', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Dropdown(
                        id='optuna-features',
                        options=[{'label': 'ALL FEATURES', 'value': 'ALL FEATURES'}] + features,
                        multi=True,
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '20px'}),
                    html.B('Number of iterations', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(dcc.Slider(
                        id='optuna-n-iterations',
                        min=1,
                        max=100,
                        step=1,
                        value=10,
                        marks=None,
                        tooltip={'always_visible': False, 'placement': 'bottom'},
                    ), style={'width': '100%', 'margin-top': '12px'}),
                ], style={'display': 'flex'}),
                html.Div(style={'height': '5px'}),
                html.Div([
                    html.B('Estimator', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Dropdown(
                        id='optuna-estimator',
                        options=[
                            {'label': 'Logistic Regression', 'value': 'logreg'},
                            {'label': 'Random Forest', 'value': 'rf'},
                            {'label': 'XGBoost', 'value': 'xgb'},
                        ],
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '20px'}),
                    html.B('Scoring', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Dropdown(
                        id='optuna-scoring',
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
                        id='optuna-cv',
                        options=[
                            {'label': 'KFold', 'value': 'kfold'},
                            {'label': 'StratifiedKFold', 'value': 'skfold'},
                            {'label': 'TimeSeriesSplit', 'value': 'tssplit'},
                            {'label': 'CPCV', 'value': 'cpcv'},
                        ],
                        style={'width': '100%'}
                    ),
                ], style={'display': 'flex'}),
                html.Div(style={'height': '10px'}),
                dbc.Button(
                    "Run Search",
                    id="optuna-run",
                    color="primary",
                    style={'width': '100%'}
                )
            ]
        return [
            html.Div([
                html.B('Target', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                html.Div(style={'width': '10px'}),
                dcc.Dropdown(
                    id='optuna-target',
                    options=features,
                    style={'width': '100%'}
                ),
                html.Div(style={'width': '10px'}),
                html.B('Features', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                html.Div(style={'width': '10px'}),
                dcc.Dropdown(
                    id='optuna-features',
                    options=[{'label': 'ALL FEATURES', 'value': 'ALL FEATURES'}] + features,
                    multi=True,
                    style={'width': '100%'}
                ),
                html.Div(style={'width': '20px'}),
                html.B('Number of iterations', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                html.Div(dcc.Slider(
                    id='optuna-n-iterations',
                    min=1,
                    max=100,
                    step=1,
                    value=10,
                    marks=None,
                    tooltip={'always_visible': False, 'placement': 'bottom'},
                ), style={'width': '100%', 'margin-top': '12px'}),
            ], style={'display': 'flex'}),
            html.Div(style={'height': '5px'}),
            html.Div([
                html.B('Estimator', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                html.Div(style={'width': '10px'}),
                dcc.Dropdown(
                    id='optuna-estimator',
                    options=[
                        {'label': 'Linear Regression', 'value': 'linreg'},
                        {'label': 'Random Forest', 'value': 'rf'},
                        {'label': 'XGBoost', 'value': 'xgb'},
                    ],
                    style={'width': '100%'}
                ),
                html.Div(style={'width': '20px'}),
                html.B('Scoring', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                html.Div(style={'width': '10px'}),
                dcc.Dropdown(
                    id='optuna-scoring',
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
                    id='optuna-cv',
                    options=[
                        {'label': 'KFold', 'value': 'kfold'},
                        {'label': 'TimeSeriesSplit', 'value': 'tssplit'},
                        {'label': 'CPCV', 'value': 'cpcv'},
                    ],
                    style={'width': '100%'}
                ),
            ], style={'display': 'flex'}),
            html.Div(style={'height': '10px'}),
            dbc.Button(
                "Run Search",
                id="optuna-run",
                color="primary",
                style={'width': '100%'}
            )
        ]
    raise PreventUpdate

@app.callback(
    Output('optuna-features', 'options'),
    Input('optuna-target', 'value'),
    State('the-data', 'data')
)
def update_rand_search_features(target, data):
    return [{'label': 'ALL FEATURES', 'value': 'ALL FEATURES'}] + [{'label':i, 'value':i} for i in data[0].keys() if i != target]


# Helper Methods
def isBinary(series):
    if not np.issubdtype(series.dtype, np.number):
        raise TypeError('Series must be numeric')
    if not isinstance(series, pd.Series):
        raise TypeError('Series must be of type pd.Series')
    
    if series.nunique() == 2:
        return True
    else:
        return False

