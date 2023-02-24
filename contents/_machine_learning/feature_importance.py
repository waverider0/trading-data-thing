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
    if model_type:
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
                        {'label': 'Time Series Split', 'value': 'tss'},
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

#def render_sfs_inputs(model_type, data):

@app.callback(
    Output('rfe-plot-container', 'children'),
    Input('rfe-run', 'n_clicks'),
    State('rfe-model-type', 'value'),
    State('rfe-target', 'value'),
    State('rfe-estimator', 'value'),
    State('rfe-scoring', 'value'),
    State('rfe-cv', 'value'),
    State('the-data', 'data'),
)
def run_rfe(n_clicks, model_type, target, estimator, scoring_metric, cv, data):
    if n_clicks and model_type and target and estimator and scoring_metric and cv and data:
        clf_mappings = {
            'lr': LogisticRegression(),
            'rf': RandomForestClassifier(),
            'xgb': XGBClassifier()
        }
        reg_mappings = {
            'lr': LinearRegression(),
            'rf': RandomForestRegressor(),
            'xgb': XGBRegressor()
        }
        cv_mappings = {
            'kfold': KFold(),
            'skfold': StratifiedKFold(),
            'tss': TimeSeriesSplit(),
        }
        df = pd.DataFrame.from_dict(data)
        df = df.iloc[:int(len(df) * 0.75)]
        X = df.drop(target, axis=1)
        y = df[target]

        scoring = None
        rfecv = None
        if model_type == 'clf':
            y = pd.Series(LabelEncoder().fit_transform(y))
            if isBinary(y): # binary classification
                scoring_mappings = {
                    'accuracy': make_scorer(accuracy_score),
                    'precision': make_scorer(precision_score),
                    'recall': make_scorer(recall_score),
                    'f1': make_scorer(f1_score),
                }
                scoring = scoring_mappings[scoring_metric]
            else: # multi-class classification
                scoring_mappings = {
                    'accuracy': make_scorer(accuracy_score),
                    'precision': make_scorer(precision_score, average='weighted'),
                    'recall': make_scorer(recall_score, average='weighted'),
                    'f1': make_scorer(f1_score, average='weighted'),
                }
                scoring = scoring_mappings[scoring_metric]
            rfecv = RFECV(
                estimator=clf_mappings[estimator],
                scoring=scoring,
                cv=cv_mappings[cv],
                n_jobs=-1
            )
            rfecv.fit(X, y)
        else:
            if cv == 'skfold': cv = 'kfold' # skfold doesn't work one regression problems
            scoring_mappings = {
                'r2': make_scorer(r2_score),
                'mse': make_scorer(mean_squared_error, greater_is_better=False),
                'rmse': make_scorer(mean_squared_error, greater_is_better=False, squared=False),
                'mae': make_scorer(mean_absolute_error, greater_is_better=False),
            }
            scoring = scoring_mappings[scoring_metric]
            rfecv = RFECV(
                estimator=reg_mappings[estimator],
                scoring=scoring,
                cv=cv_mappings[cv],
                n_jobs=-1
            )
            rfecv.fit(X, y)

        return [
            dcc.Graph(figure=go.Figure(
                go.Scatter(
                    x=list(range(1, len(rfecv.cv_results_['mean_test_score']) + 1)),
                    y=rfecv.cv_results_['mean_test_score'],
                    error_y=dict(
                        type='data',
                        array=rfecv.cv_results_['std_test_score'],
                        thickness=0.5,
                    ),
                    marker=dict(color='#37699b'),
                ),
                layout=go.Layout(
                    title='RFE Results',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#FFFFFF'),
                    xaxis=dict(
                        title='Number of Features',
                        titlefont=dict(color='#FFFFFF'),
                        showgrid=False,
                    ),
                    yaxis=dict(
                        title='Score',
                        titlefont=dict(color='#FFFFFF'),
                        showgrid=False,
                    ),
                )
            )),
        ]

    raise PreventUpdate

@app.callback(
    Output('boruta-shap-plot-container', 'children'),
    Input('boruta-shap-run', 'n_clicks'),
    State('boruta-shap-model-type', 'value'),
    State('boruta-shap-target', 'value'),
    State('boruta-shap-n-trials', 'value'),
    State('the-data', 'data'),
)
def run_boruta_shap(n_clicks, model_type, target, n_trials, data):
    if n_clicks and model_type and target and n_trials and data:
        df = pd.DataFrame.from_dict(data)
        df = df.iloc[:int(len(df) * 0.75)]
        X = df.drop(target, axis=1)
        y = df[target]
        Feature_Selector = BorutaShap(importance_measure='shap', classification=True if model_type == 'clf' else False)
        Feature_Selector.fit(
            X=X,
            y=y,
            n_trials=n_trials,
            sample=False,
            train_or_test='test',
            normalize=True,
            verbose=True
        )
        box_plot_components = Feature_Selector.return_plot_components(which_features='all')
        box_plot_components = box_plot_components.sort_values('Z-Score', ascending=False)

        return [
            dcc.Graph(
                figure=go.Figure(
                    go.Box(
                        x=box_plot_components['Features'],
                        y=box_plot_components['Z-Score'],
                        marker=dict(color='#00ff00'),
                    ),
                    layout=go.Layout(
                        title='Boruta Shap Results',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#FFFFFF'),
                        xaxis=dict(
                            title='Feature',
                            titlefont=dict(color='#FFFFFF'),
                            showgrid=False,
                        ),
                        yaxis=dict(
                            title='Importance',
                            titlefont=dict(color='#FFFFFF'),
                            showgrid=False,
                        ),
                    )
                ),
            ),
        ]
    
    raise PreventUpdate


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
