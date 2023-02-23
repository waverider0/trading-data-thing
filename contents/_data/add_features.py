import numpy as np
import pandas as pd

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

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
                    df[f'{base_feature} Percent Change'] = df[base_feature].pct_change(input)
                elif transformation == 'log_ret':
                    df[f'{base_feature} Log Return'] = np.log(df[base_feature]) - np.log(df[base_feature].shift(input))
                elif transformation == 'min':
                    df[f'Min{input}({base_feature})'] = df[base_feature].rolling(input).min()
                elif transformation == 'max':
                    df[f'Max{input}({base_feature})'] = df[base_feature].rolling(input).max()
                elif transformation == 'min_pos':
                    df[f'MinPos{input}({base_feature})'] = df[base_feature].rolling(input).apply(lambda x: np.argmin(x) + 1)
                elif transformation == 'max_pos':
                    df[f'MaxPos{input}({base_feature})'] = df[base_feature].rolling(input).apply(lambda x: np.argmax(x) + 1)
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

