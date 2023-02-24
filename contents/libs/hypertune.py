from functools import partial
import random

import numpy as np
import pandas as pd
import optuna

from .cross_validation import cv_score

class RandomSearch:
    def __init__(self, model, cv_method, scorer, greater_is_better, param_grid, n_iter):
        """
        Initialize the RandomSearch class

        Parameters:
            - model: An instance of the ML model to be tuned
            - cv_method (str): The cross validation method to use
            - scorer (sklearn.metrics): make_scorer() scorer object
            - greater_is_better (bool): Whether a higher score is better
            - param_grid (dict): A dictionary of parameters and the range of possible values
            - n_iter (int): Number of iterations for the tuning process
        """
        self.model = model
        self.cv_method = cv_method
        self.scorer = scorer
        self.greater_is_better = greater_is_better
        self.param_grid = param_grid
        self.n_iter = n_iter

    def tune(self, X, y):
        """
        Perform the random search tuning process

        Parameters:
            - X (pd.DataFrame): Feature values
            - y (pd.Series): Target values
        
        Returns:
            - results (pd.DataFrame): A dataframe containing the results
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError('X must be a pandas dataframe')
        if not isinstance(y, pd.Series):
            raise TypeError('y must be a pandas series')

        results = {'params':[], 'train_score':[], 'test_score':[]}
        for i in range(self.n_iter):
            # Sample random hyperparameters
            curr_params = {param: random.choice(values) for param, values in self.param_grid.items()}
            
            # Set the hyperparameters for the model
            self.model.set_params(**curr_params)
            
            # Compute scores and store results
            try: train_scores, test_scores = cv_score(X, y, self.model, self.scorer, method=self.cv_method)
            except: continue
            results['params'] = results['params'] + [curr_params]
            if self.greater_is_better: # If a higher score is better, subtract the standard deviation
                results['train_score'] = results['train_score'] + [np.mean(train_scores) - np.std(train_scores)]
                results['test_score'] = results['test_score'] + [np.mean(test_scores) - np.std(test_scores)]
            else: # If a lower score is better, add the standard deviation
                results['train_score'] = results['train_score'] + [np.mean(train_scores) + np.std(train_scores)]
                results['test_score'] = results['test_score'] + [np.mean(test_scores) + np.std(test_scores)]
        
        # Convert results to a dataframe and return
        df = pd.DataFrame(results)
        df = pd.concat([df.drop(['params'], axis=1), df['params'].apply(pd.Series)], axis=1)
        return df

class OptunaSearch:
    def __init__(self, model, cv_method, scorer, greater_is_better, param_grid, n_iter):
        """
        Initialize the OptunaSearch class

        Parameters:
            - model: An instance of the ML model to be tuned
            - cv_method (str): The cross validation method to use
            - scorer (sklearn.metrics): make_scorer() scorer object
            - greater_is_better (bool): Whether a higher score is better
            - param_grid (dict): A dictionary of parameters and the range of possible values
            - n_iter (int): Number of iterations for the tuning process
        """
        self.model = model
        self.cv_method = cv_method
        self.scorer = scorer
        self.greater_is_better = greater_is_better
        self.param_grid = param_grid
        self.n_iter = n_iter

        self.results = {'params':[], 'train_score':[], 'test_score':[]}

    def tune(self, X, y):
        objective_function = partial(
            self.objective,
            param_grid=self.param_grid,
            estimator=self.model,
            X=X,
            y=y
        )
        # delete any existing study before creating a new one
        try: optuna.delete_study(study_name='optuna_search')
        except: pass
        study = optuna.create_study(study_name='optuna_search', direction='maximize' if self.greater_is_better else 'minimize')
        study.optimize(objective_function, n_trials=self.n_iter)
        
        # Convert results to a dataframe and return
        df = pd.DataFrame(self.results)
        df = pd.concat([df.drop(['params'], axis=1), df['params'].apply(pd.Series)], axis=1)
        return df

    def objective(self, trial, param_grid, estimator, X, y):
        params = {param: trial.suggest_categorical(param, values) for param, values in param_grid.items()}
        estimator.set_params(**params)

        try:
            train_scores, test_scores = cv_score(X, y, estimator, self.scorer, method=self.cv_method)
            self.results['params'] = self.results['params'] + [params]
            if self.greater_is_better: # If a higher score is better, subtract the standard deviation
                self.results['train_score'] = self.results['train_score'] + [np.mean(train_scores) - np.std(train_scores)]
                self.results['test_score'] = self.results['test_score'] + [np.mean(test_scores) - np.std(test_scores)]
            else: # If a lower score is better, add the standard deviation
                self.results['train_score'] = self.results['train_score'] + [np.mean(train_scores) + np.std(train_scores)]
                self.results['test_score'] = self.results['test_score'] + [np.mean(test_scores) + np.std(test_scores)]
            return np.mean(self.results['test_score'][-1])
        except:
            pass