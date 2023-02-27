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
        if feature == 'index':
            df = pd.DataFrame.from_dict(data)
            df = df.sort_index(ascending=order=='ascending')
            return df.to_dict('records'), ''
        else:
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

@app.callback(
    Output('the-data', 'data'),
    Output('hidden-div', 'children'),
    Input('add-ohlcv-features-button', 'n_clicks'),
    State('open-feature-dropdown', 'value'),
    State('high-feature-dropdown', 'value'),
    State('low-feature-dropdown', 'value'),
    State('close-feature-dropdown', 'value'),
    State('volume-feature-dropdown', 'value'),
    State('ohlcv-features-dropdown', 'value'),
    State('ohlcv-features-window', 'value'),
    State('the-data', 'data'),
)
def add_ohlcv_features(n_clicks, open, high, low, close, volume, features, window, data):
    if n_clicks and open and high and low and close and volume and features and window and data:
        df = pd.DataFrame.from_dict(data)
        for feature in features:
            if feature == 'rsi':
                pass
            elif feature == 'atr':
                pass
            elif feature == 'c2c_vol':
                df[f'C2CVol{window}'] = (np.log(df[close]) - np.log(df[close].shift(1))).rolling(window).std()
            elif feature == 'parkinson_vol':
                h_l = np.log(df[high] / df[low]) ** 2
                df[f'ParkinsonVol{window}'] = h_l.rolling(window).apply(
                    lambda x: np.sqrt((1 / (4 * window * np.log(2))) * np.sum(x))
                )
            elif feature == 'garman_klass_vol':
                h_l = (
                    (np.log(df[high] / df[low]) ** 2) / 2
                ).rolling(window).mean()
                c = (
                    2 * np.log(2) - 1) * (np.log(df[close] / df[close].shift()) ** 2
                ).rolling(window).mean()
                df[f'GarmanKlassVol{window}'] = np.sqrt(h_l - c)
            elif feature == 'rodgers_satchell_vol':
                x = (
                    (np.log(df[high] / df[close]) * np.log(df[high] / df[open])) +\
                    (np.log(df[low] / df[close]) * np.log(df[low] / df[open]))
                )
                df[f'RodgersSatchellVol{window}'] = np.sqrt(x.rolling(window).mean())
            elif feature == 'yang_zhang_vol':
                o = ((
                    np.log(df[open] / df[open].shift()) -\
                    np.log(df[open] / df[open].shift()).rolling(window).mean()
                ) ** 2).rolling(window).apply(lambda x: (1 / (window -  1)) * np.sum(x))
                c = ((
                    np.log(df[close] / df[close].shift()) -\
                    np.log(df[close] / df[close].shift()).rolling(window).mean()
                ) ** 2).rolling(window).apply(lambda x: (1 / (window -  1)) * np.sum(x))
                rs = (
                    (np.log(df[high] / df[close]) * np.log(df[high] / df[open])) +\
                    (np.log(df[low] / df[close]) * np.log(df[low] / df[open]))
                ).rolling(window).mean()
                k = 0.34 / (1.34 + (window + 1) / (window - 1))
                df[f'YangZhangVol{window}'] = np.sqrt(o + k * c + (1 - k) * rs)
        return df.to_dict('records'), ''
    raise PreventUpdate


# Helper Methods
