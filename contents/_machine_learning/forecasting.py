import os
import shutil
import pickle

from dash import dcc, html, ALL
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objs as go

import math
import pandas as pd
import numpy as np

from contents.app import *
from contents.storage.utils import get_saved_models


@app.callback(
    Output('forecast-features-container', 'children'),
    Output('forecast-results', 'children'),
    Input('pre-compiled-models', 'value'),
)
def update_forecast_inputs(model_name):
    if model_name:
        try:  _, features = load_model(model_name)
        except: return '', ''

        children = [
            html.B('Feature Inputs'),
            html.Br(),
        ]
        for feature in features:
            children.append(html.Div([
                dcc.Input(
                    id={
                        'type': 'forecast-features',
                        'index': feature
                    },
                    type='number',
                    placeholder=feature,
                    debounce=True,
                    style={'width': '100%'}
                ),
                html.Div(style={'width': '10px'}),
            ], style={'display': 'inline-block', 'width': '50%'}))

        return children, ''
            
    raise PreventUpdate

@app.callback(
    Output('forecast-results', 'children'),
    Input('generate-forecast', 'n_clicks'),
    State('pre-compiled-models', 'value'),
    State({'type': 'forecast-features', 'index': ALL}, 'value'),
)
def generate_forecast(n_clicks, model_name, feature_values):
    if n_clicks and model_name and feature_values:
        model, _ = load_model(model_name)
        X = np.array(feature_values).reshape(1, -1)

        children = []
        forecast = model.predict(X)
        children.append(html.Div([
            html.B('Forecast:'),
            html.Div(style={'width': '10px'}),
            html.Div(forecast)
        ], style={'display': 'flex'}))
        try:
            children.append(html.Div([
                html.B('Probability:'),
                html.Div(style={'width': '10px'}),
                html.Div(model.predict_proba(X)[0][forecast])
            ], style={'display': 'flex'}))
        except:
            pass

        return children

    raise PreventUpdate

@app.callback(
    Output('pre-compiled-models', 'options'),
    Output('forecast-results', 'children'),
    Input('delete-model', 'n_clicks'),
    State('pre-compiled-models', 'value')
)
def delete_selected_model(n_clicks, selected_model):
    if n_clicks and selected_model:
        delete_model(selected_model)
        return get_saved_models(), ''
    raise PreventUpdate


# Helper Methods
def load_model(model_name):
    """
        Loads a model from the saved_models directory
        and returns the model and the features used to train it
    """
    dirname = os.path.dirname(os.path.dirname(__file__))

    with open(f'{dirname}/storage/saved_models/{model_name}/compiled_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open(f'{dirname}/storage/saved_models/{model_name}/features.txt', 'r') as f:
        features = f.read().split(',')

    return model, features

def delete_model(model_name):
    """ Deletes a model from the saved_models directory """
    dirname = os.path.dirname(os.path.dirname(__file__))
    shutil.rmtree(f'{dirname}/storage/saved_models/{model_name}')