import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from contents.app import *


@app.callback(
    Output('var-plot-container', 'children'),
    Input('var-plot-type', 'value'),
    Input('var-plot-scale-data', 'on'),
    State('the-data', 'data')
)
def render_var_plot(plot_type, scale_data, data):
    if plot_type and data:
        df = pd.DataFrame.from_dict(data)
        if scale_data: df = normalize_df(df)

        if plot_type == 'box':
            return [
                dcc.Graph(
                    figure={
                        'data': [go.Box(y=df[col], name=col, boxpoints='outliers') for col in df.columns],
                        'layout': go.Layout(
                            title='Feature Box Plot',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#FFFFFF'),
                            xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=False),
                        ),
                    }
                ),
            ]
        elif plot_type == 'violin':
            return [
                dcc.Graph(
                    figure={
                        'data': [go.Violin(y=df[col], name=col, points='outliers', meanline_visible=True) for col in df.columns],
                        'layout': go.Layout(
                            title='Feature Violin Plot',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#FFFFFF'),
                            xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=False),
                        ),
                    }
                ),
            ]
        
    raise PreventUpdate

@app.callback(
    Output('dist-plot-container', 'children'),
    Input('dist-plot-scale-data', 'on'),
    Input('dist-plot-feature', 'value'),
    Input('dist-plot-distributions', 'value'),
    State('the-data', 'data'),
)
def render_dist_plot(scale_data, feature, distributions, data):
    if feature and distributions and data:
        df = pd.DataFrame.from_dict(data)
        if scale_data: df = normalize_df(df)

        graph = dcc.Graph(
            figure=go.Figure(
                layout=go.Layout(
                    title='Empirical vs Theoretical Distributions',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#FFFFFF'),
                    xaxis=dict(
                        title=feature,
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
        graph.figure.add_trace(
            go.Histogram(
                x=df[feature],
                name=feature,
                histnorm='probability density',
                marker=dict(color='#37699b'),
            )
        )
        for dist in distributions:
            if dist == 'normal':
                mu, std = norm.fit(df[feature])
                x = np.linspace(min(df[feature]), max(df[feature]), 100)
                p = norm.pdf(x, mu, std)
                graph.figure.add_trace(
                    go.Scatter(
                        x=x,
                        y=p,
                        mode='lines',
                        name='Normal',
                    )
                )
            elif dist == 'lognormal':
                pass
        
        return graph

    raise PreventUpdate


# Helper Methods
def normalize_df(df):
    """ Normalize a dataframe with MinMaxScaler (Keep the column names) """
    cols = df.columns
    df = pd.DataFrame(MinMaxScaler().fit_transform(df))
    df.columns = cols
    return df