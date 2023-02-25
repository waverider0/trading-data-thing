import os
import shutil

import pandas as pd


# Set dirname to folder containing this file
dirname = os.path.dirname(__file__)

def delete_temp_models(model_name):
    if os.listdir(f'{dirname}/temp'):
        if os.path.exists(f'{dirname}/temp/{model_name}'):
            shutil.rmtree(f'{dirname}/temp/{model_name}')

def get_saved_models():
    """ Returns a list of all the saved models """
    saved_models = []
    if os.listdir(f'{dirname}/saved_models'):
        for file in os.listdir(f'{dirname}/saved_models'):
            saved_models.append(file)
    return saved_models