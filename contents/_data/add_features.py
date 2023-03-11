import numpy as np
import pandas as pd

from dash import ALL
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import StackingClassifier, StackingRegressor

from contents.app import *


@app.callback(
    Output('the-data', 'data'),
    Output('hidden-div', 'children'),
    Input('drop-features-button', 'n_clicks'),
    State('drop-features-dropdown', 'value'),
    State('the-data', 'data'),
)
def drop_features(n_clicks, features, data):
    df = pd.DataFrame.from_dict(data)
    if n_clicks and features and data:
        df = df.drop(features, axis=1)
        return df.to_dict('records'), ''
    raise PreventUpdate

def drop_rows():
    pass

@app.callback(
    Output('the-data', 'data'),
    Output('hidden-div', 'children'),
    Input('rename-feature-button', 'n_clicks'),
    State('rename-features-dropdown', 'value'),
    State('feature-new-name', 'value'),
    State('the-data', 'data'),
)
def rename_feature(n_clicks, feature, new_name, data):
    if n_clicks and feature and new_name and data:
        df = pd.DataFrame.from_dict(data)
        df = df.rename(columns={feature: new_name})
        return df.to_dict('records'), ''
    raise PreventUpdate

@app.callback(
    Output('the-data', 'data'),
    Output('hidden-div', 'children'),
    Input('order-by-apply-button', 'n_clicks'),
    State('order-by-feature-dropdown', 'value'),
    State('order-by-order-dropdown', 'value'),
    State('the-data', 'data')
)
def order_by(n_clicks, feature, order, data):
    if n_clicks and feature and order and data:
        df = pd.DataFrame.from_dict(data)
        df = df.sort_values(feature, ascending=order=='ascending')
        return df.to_dict('records'), ''
    raise PreventUpdate

@app.callback(
    Output('the-data', 'data'),
    Output('hidden-div', 'children'),
    Input('add-features-button', 'n_clicks'),
    State('transformations-dropdown', 'value'),
    State('transformations-input', 'value'),
    State('transformations-base-features', 'value'),
    State('the-data', 'data'),
)
def add_transformation(n_clicks, transformations, input, base_features, data):
    if n_clicks and transformations and input and base_features and data:
        df = pd.DataFrame.from_dict(data)
        for transformation in transformations:
            for base_feature in base_features:
                if transformation == 'nth_power':
                    df[f'{base_feature}^({input})'] = df[base_feature] ** input
                elif transformation == 'nth_root':
                    df[f'{base_feature}^({round(1 / input, 2)})'] = df[base_feature] ** (1 / input)
                elif transformation == 'log_n':
                    df[f'Log{input}({base_feature})'] = np.log(df[base_feature]) / np.log(input)
                elif transformation == 'abs':
                    df[f'Abs({base_feature})'] = np.abs(df[base_feature])
                elif transformation == 'pct_change':
                    df[f'PctChange{input}({base_feature})'] = df[base_feature].pct_change(input)
                elif transformation == 'log_ret':
                    df[f'LogRet{input}({base_feature})'] = np.log(df[base_feature]) - np.log(df[base_feature].shift(input))
                elif transformation == 'min':
                    df[f'Min{input}({base_feature})'] = df[base_feature].rolling(input).min()
                elif transformation == 'max':
                    df[f'Max{input}({base_feature})'] = df[base_feature].rolling(input).max()
                elif transformation == 'min_pos':
                    df[f'MinPos{input}({base_feature})'] = input - df[base_feature].rolling(input).apply(lambda x: np.argmin(x) + 1)
                elif transformation == 'max_pos':
                    df[f'MaxPos{input}({base_feature})'] = input - df[base_feature].rolling(input).apply(lambda x: np.argmax(x) + 1)
                elif transformation == 'sum':
                    df[f'Sum{input}({base_feature})'] = df[base_feature].rolling(input).sum()
                elif transformation == 'sma':
                    df[f'SMA{input}({base_feature})'] = df[base_feature].rolling(input).mean()
                elif transformation == 'ema':
                    df[f'EMA{input}({base_feature})'] = df[base_feature].ewm(span=input, adjust=False).mean()
                elif transformation == 'std':
                    df[f'Std{input}({base_feature})'] = df[base_feature].rolling(input).std()
                elif transformation == 'var':
                    df[f'Var{input}({base_feature})'] = df[base_feature].rolling(input).var()
                elif transformation == 'skew':
                    df[f'Skew{input}({base_feature})'] = df[base_feature].rolling(input).skew()
                elif transformation == 'kurt':
                    df[f'Kurt{input}({base_feature})'] = df[base_feature].rolling(input).kurt()
                elif transformation == 'z_score':
                    df[f'ZScore{input}({base_feature})'] = (df[base_feature] - df[base_feature].rolling(input).mean()) / df[base_feature].rolling(input).std()
                elif transformation == 'lag':
                    df[f'Lag{input}({base_feature})'] = df[base_feature].shift(input)
                elif transformation == 'nth_deriv':
                    nth_deriv = df[base_feature]
                    for n in range(input): nth_deriv = nth_deriv.diff()
                    df[f'{base_feature}^{input}th Derivative'] = nth_deriv
                elif transformation == 'corr':
                    if len(base_features) > 1:
                        for base_feature_2 in base_features:
                            if base_feature != base_feature_2:
                                if f'Corr{input}({base_feature_2}, {base_feature})' not in df.columns:
                                    df[f'Corr{input}({base_feature}, {base_feature_2})'] = df[base_feature].rolling(input).corr(df[base_feature_2])
        return df.to_dict('records'), ''

    raise PreventUpdate

##############
# METALABELS #
##############
""" @app.callback(
    Output('metalabels-clf-features', 'options'),
    Input('metalabels-clf-target', 'value'),
    State('the-data', 'data')
)
def update_clf_features(target, data):
    return [{'label': 'ALL FEATURES', 'value': 'ALL FEATURES'}] + [{'label':i, 'value':i} for i in data[0].keys() if i != target]

@app.callback(
    Output('metalabels-reg-features', 'options'),
    Input('metalabels-reg-target', 'value'),
    State('the-data', 'data')
)
def update_reg_features(target, data):
    return [{'label': 'ALL FEATURES', 'value': 'ALL FEATURES'}] + [{'label':i, 'value':i} for i in data[0].keys() if i != target] """

@app.callback(
    Output('metalabels-model-inputs', 'children'),
    Input('metalabels-model-type', 'value'),
    State('the-data', 'data'),
)
def render_metalabels_model_inputs(model_type, data):
    if model_type and data:
        df = pd.DataFrame.from_dict(data)
        features = [{'label': feature, 'value': feature} for feature in df.columns]
        if model_type == 'clf':
            return [
                html.Div([
                    html.B('Target', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Dropdown(
                        id='metalabels-clf-target',
                        options=features,
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '20px'}),
                    html.B('Features', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Dropdown(
                        id='metalabels-clf-features',
                        options=features,
                        multi=True,
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '20px'}),
                    html.B('Estimators', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Dropdown(
                        id='metalabels-clf-estimators',
                        options=[
                            {'label': 'Logistic Regression', 'value': 'lr_clf'},
                            {'label': 'Random Forest', 'value': 'rf_clf'},
                            {'label': 'XGBoost', 'value': 'xgb_clf'},
                        ],
                        multi=True,
                        style={'width': '100%'}
                    ),
                    html.Div(style={'width': '20px'}),
                    html.B('Window', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                    html.Div(style={'width': '10px'}),
                    dcc.Input(
                        id='metalabels-clf-window',
                        type='number',
                        style={'width': '100%'}
                    ),
                ], style={'display': 'flex'}),
                html.Div(style={'height': '10px'}),
                html.Div(id='metalabels-clf-hyperparam-inputs-container'),
                html.Hr(),
                dbc.Button(
                    'Generate Metalabels',
                    id='clf-generate-metalabels-button',
                    color='primary',
                    style={'width': '100%'}
                ),
            ]
        return [
            html.Div([
                html.B('Target', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                html.Div(style={'width': '10px'}),
                dcc.Dropdown(
                    id='metalabels-reg-target',
                    options=features,
                    style={'width': '100%'}
                ),
                html.Div(style={'width': '20px'}),
                html.B('Features', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                html.Div(style={'width': '10px'}),
                dcc.Dropdown(
                    id='metalabels-reg-features',
                    options=features,
                    multi=True,
                    style={'width': '100%'}
                ),
                html.Div(style={'width': '20px'}),
                html.B('Estimators', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                html.Div(style={'width': '10px'}),
                dcc.Dropdown(
                    id='metalabels-reg-estimators',
                    options=[
                        {'label': 'Linear Regression', 'value': 'lr_reg'},
                        {'label': 'Random Forest', 'value': 'rf_reg'},
                        {'label': 'XGBoost', 'value': 'xgb_reg'},
                    ],
                    multi=True,
                    style={'width': '100%'}
                ),
                html.B('Window', style={'margin-top': '5px', 'white-space': 'nowrap'}),
                html.Div(style={'width': '10px'}),
                dcc.Input(
                    id='metalabels-reg-window',
                    type='number',
                    style={'width': '100%'}
                ),
            ], style={'display': 'flex'}),
            html.Div(style={'height': '10px'}),
            html.Div(id='metalabels-reg-hyperparam-inputs-container'),
            html.Hr(),
            dbc.Button(
                'Generate Metalabels',
                id='reg-generate-metalabels-button',
                color='primary',
                style={'width': '100%'}
            ),
        ]
    raise PreventUpdate

@app.callback(
    Output('metalabels-clf-hyperparam-inputs-container', 'children'),
    Input('metalabels-clf-estimators', 'value')
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
                            'metalabeler': estimator,
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
                            'metalabeler': estimator,
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
                            'metalabeler': estimator,
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
                            'metalabeler': estimator,
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
                            'metalabeler': estimator,
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
                            'metalabeler': estimator,
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
                            'metalabeler': estimator,
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
                            'metalabeler': estimator,
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
                            'metalabeler': estimator,
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
                            'metalabeler': estimator,
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
                            'metalabeler': estimator,
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
                            'metalabeler': estimator,
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
                            'metalabeler': estimator,
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
                            'metalabeler': estimator,
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
                            'metalabeler': estimator,
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
                            'metalabeler': estimator,
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
    Output('metalabels-reg-hyperparam-inputs-container', 'children'),
    Input('metalabels-reg-estimators', 'value')
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
                            'metalabeler': estimator,
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
                            'metalabeler': estimator,
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
                            'metalabeler': estimator,
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
                            'metalabeler': estimator,
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
                            'metalabeler': estimator,
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
                            'metalabeler': estimator,
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
                            'metalabeler': estimator,
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
                            'metalabeler': estimator,
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
                            'metalabeler': estimator,
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
                            'metalabeler': estimator,
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
                            'metalabeler': estimator,
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
                            'metalabeler': estimator,
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
                            'metalabeler': estimator,
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
                            'metalabeler': estimator,
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
    Output('the-data', 'data'),
    Output('hidden-div', 'children'),
    Input('clf-generate-metalabels-button', 'n_clicks'),
    State('metalabels-clf-target', 'value'),
    State('metalabels-clf-features', 'value'),
    State('metalabels-clf-estimators', 'value'),
    State('metalabels-clf-window', 'value'),
    State({'metalabeler': 'lr_clf', 'param': ALL}, 'value'),
    State({'metalabeler': 'rf_clf', 'param': ALL}, 'value'),
    State({'metalabeler': 'xgb_clf', 'param': ALL}, 'value'),
    State('the-data', 'data')
)
def generate_clf_metalabels(
    n_clicks,
    target,
    features,
    estimators,
    window,
    lr_params,
    rf_params,
    xgb_params,
    data
):
    if (
        n_clicks and
        target and
        features and
        estimators and
        window and
        data
    ):
        # if features == ['ALL FEATURES']: features = [col for col in data[0].keys() if col != target]
        # elif len(features) > 1 and 'ALL FEATURES' in features: features.remove('ALL FEATURES')
        df = pd.DataFrame.from_dict(data)
        
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
        else: # single model
            model = estimator_mappings[estimators[0]]

        # Generate metalabels
        df[f'ClfMetalabelerSuccess({target})'] = np.nan
        for i in range(window, len(df)):
            model.fit(df[features].iloc[i-window:i], df[target].iloc[i-window:i])
            # 1 if the model predicts the correct class, 0 otherwise
            df.loc[i, f'ClfMetalabelerSuccess({target})'] = 1 if model.predict(df[features].iloc[i].values.reshape(1, -1))[0] == df[target].iloc[i] else 0
       
        return df.to_dict('records'), ''

    raise PreventUpdate
        
@app.callback(
    Output('the-data', 'data'),
    Output('hidden-div', 'children'),
    Input('reg-generate-metalabels-button', 'n_clicks'),
    State('metalabels-reg-target', 'value'),
    State('metalabels-reg-features', 'value'),
    State('metalabels-reg-estimators', 'value'),
    State('metalabels-reg-window', 'value'),
    State({'metalabeler': 'lr_reg', 'param': ALL}, 'value'),
    State({'metalabeler': 'rf_reg', 'param': ALL}, 'value'),
    State({'metalabeler': 'xgb_reg', 'param': ALL}, 'value'),
    State('the-data', 'data')
)
def generate_reg_metalabels(
    n_clicks,
    target,
    features,
    estimators,
    window,
    lr_params,
    rf_params,
    xgb_params,
    data
):
    if (
        n_clicks and
        target and
        features and
        estimators and
        window and
        data
    ):
        # if features == ['ALL FEATURES']: features = [col for col in data[0].keys() if col != target]
        # elif len(features) > 1 and 'ALL FEATURES' in features: features.remove('ALL FEATURES')
        df = pd.DataFrame.from_dict(data)
        
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

        # Generate metalabels
        df[f'RegMetalabelerError({target})'] = np.nan
        for i in range(window, len(df)):
            model.fit(df[features].iloc[i-window:i], df[target].iloc[i-window:i])
            df.loc[i, f'RegMetalabelerError({target})'] = df.loc[i, target] - model.predict(df[features].iloc[i].values.reshape(1, -1))[0]

        return df.to_dict('records'), ''
    
    raise PreventUpdate

# Helper Methods
