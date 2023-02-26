import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, cramervonmises_2samp

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from contents.app import *


@app.callback(
    Output('drift-plot-container', 'children'),
    Input('drift-plot-test', 'value'),
    Input('drift-plot-feature', 'value'),
    Input('drift-plot-n-splits', 'value'),
    State('the-data', 'data')
)
def render_drift_plot(test, feature, n_splits, data):
    if test and feature and n_splits and data:
        df = pd.DataFrame.from_dict(data)
        splits = np.array_split(df[feature], n_splits)

        if test == 'none':
            return [
                dcc.Graph(
                    figure={
                        'data': [go.Box(y=split, name=f'Split {i}') for i, split in enumerate(splits)],
                        'layout': go.Layout(
                            title=f'{feature} Drift',
                            xaxis_title='Split',
                            yaxis_title=feature,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#FFFFFF'),
                            xaxis=dict(titlefont=dict(color='#FFFFFF'), showgrid=False),
                            yaxis=dict(titlefont=dict(color='#FFFFFF'), showgrid=False),
                        ),
                    }
                ),
            ]
        elif test == 'ks':
            ks_values = []
            for i, split in enumerate(splits):
                if i == 0: continue
                ks_values.append(ks_2samp(splits[i - 1], split).statistic)
            return [
                dcc.Graph(
                    figure={
                        'data': [go.Bar(x=list(range(1, n_splits)), y=ks_values)],
                        'layout': go.Layout(
                            title=f'{feature} Drift',
                            xaxis_title='Split',
                            yaxis_title='KS Statistic',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#FFFFFF'),
                            xaxis=dict(titlefont=dict(color='#FFFFFF'), showgrid=False),
                            yaxis=dict(titlefont=dict(color='#FFFFFF'), showgrid=False),
                        ),
                    }
                ),
            ]
        elif test == 'cvm':
            cvm_values = []
            for i, split in enumerate(splits):
                if i == 0: continue
                cvm_values.append(cramervonmises_2samp(splits[i - 1], split).statistic)
            return [
                dcc.Graph(
                    figure={
                        'data': [go.Bar(x=list(range(1, n_splits)), y=cvm_values)],
                        'layout': go.Layout(
                            title=f'{feature} Drift',
                            xaxis_title='Split',
                            yaxis_title='CVM Statistic',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#FFFFFF'),
                            xaxis=dict(titlefont=dict(color='#FFFFFF'), showgrid=False),
                            yaxis=dict(titlefont=dict(color='#FFFFFF'), showgrid=False),
                        ),
                    }
                ),
            ]

    raise PreventUpdate


