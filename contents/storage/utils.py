import os
import shutil

import pandas as pd


# Set dirname to folder containing this file
dirname = os.path.dirname(__file__)

def delete_temp_models():
    """ Deletes all of the contents of the temp folder if there are any """
    
    if os.listdir(f'{dirname}/temp'):
        for file in os.listdir(f'{dirname}/temp'):
            shutil.rmtree(f'{dirname}/temp/{file}')

def get_saved_models():
    """ Returns a list of all the saved models """
    saved_models = []
    if os.listdir(f'{dirname}/saved_models'):
        for file in os.listdir(f'{dirname}/saved_models'):
            saved_models.append(file)
    return saved_models