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
                        'type': 'forecast-model-input',
                        'index': feature
                    },
                    type='number',
                    placeholder=feature,
                    debounce=True,
                    style={'width': '100%'}
                ),
                html.Div(style={'width': '10px'}),
            ], style={'display': 'inline-block', 'width': '50%'}))
            
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