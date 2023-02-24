import math
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV

from scipy import stats

def cv_score(X, y, model, scorer, method='kfold', return_raw=False, sample_weights=None):
    """ Compute the cross validation score for a model

        :param X: (pd.DataFrame) features
        :param y: (pd.Series) target
        :param model: (sklearn model) model to use for cross validation
        :param scorer: (sklearn.metrics) make_scorer scoring function
        :param method: (str) cross validation method
        :param return_raw: (bool) return raw predictions and true values. If False, return the arrays for the train and test scores
        :param sample_weights: (pd.Series) sample weights
        :return: (list) train scores, test scores OR (dict) raw predictions and true values
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError('X must be a pandas dataframe')
    if not isinstance(y, pd.Series):
        raise TypeError('y must be a pandas series')
    if method not in ['kfold', 'skfold', 'tssplit', 'cpcv']:
        raise ValueError(f'method must be kfold, skfold, tssplit, or cpcv. Got {method}')

    train_scores = []
    test_scores = []
    preds = []
    true_vals = []
    pred_probas = []
    residuals = []
    if method == 'kfold':
        kf = KFold(n_splits=5, shuffle=True, random_state=69)
        if return_raw:
            for train_index, test_index in kf.split(X, y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                preds = preds + list(y_pred)
                true_vals = true_vals + list(y_test)
                try: pred_probas = pred_probas + list(model.predict_proba(X_test)[:, 1])
                except: pass
                try: residuals = residuals + list(y_test - y_pred)
                except: pass
            raw_output = {
                'preds': preds,
                'true_vals': true_vals,
                'pred_probas': pred_probas,
                'residuals': residuals,
            }
            return raw_output
        else:
            for train_index, test_index in kf.split(X, y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                model.fit(X_train, y_train)
                train_scores.append(scorer(model, X_train, y_train))
                test_scores.append(scorer(model, X_test, y_test))
            return train_scores, test_scores

    elif method == 'skfold':
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=69)
        if return_raw:
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                preds = preds + list(y_pred)
                true_vals = true_vals + list(y_test)
                try: pred_probas = pred_probas + list(model.predict_proba(X_test)[:, 1])
                except: pass
            raw_output = {
                'preds': preds,
                'true_vals': true_vals,
                'pred_probas': pred_probas,
                'residuals': residuals,
            }
            return raw_output
        else:
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                model.fit(X_train, y_train)
                train_scores.append(scorer(model, X_train, y_train))
                test_scores.append(scorer(model, X_test, y_test))
            return train_scores, test_scores

    elif method == 'tssplit':
        tss = TimeSeriesSplit(n_splits=5)
        if return_raw:
            for train_index, test_index in tss.split(X, y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                preds = preds + list(y_pred)
                true_vals = true_vals + list(y_test)
                try: pred_probas = pred_probas + list(model.predict_proba(X_test)[:, 1])
                except: pass
                try: residuals = residuals + list(y_test - y_pred)
                except: pass
            raw_output = {
                'preds': preds,
                'true_vals': true_vals,
                'pred_probas': pred_probas,
                'residuals': residuals,
            }
            return raw_output
        else:
            for train_index, test_index in tss.split(X, y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                model.fit(X_train, y_train)
                train_scores.append(scorer(model, X_train, y_train))
                test_scores.append(scorer(model, X_test, y_test))
            return train_scores, test_scores

    elif method == 'cpcv':
        cpcv = CombPurgedCV(N=6, k=2, purge=0, embargo=0)
        n_backtest_paths = cpcv.get_n_backtest_paths()

        # Create a dictionary to store the backtest paths
        backtest_paths = {}
        for i in range(n_backtest_paths):
            backtest_paths[i] = [[], []] # [predictions, true_values]

        # Populate the dictionary with the backtest paths
        a = 0
        b = 0
        a_start = a
        count = 0
        path_tracker = n_backtest_paths - 1
        for train, test in cpcv.split(X, y):
            model.fit(X.iloc[train], y.iloc[train])

            train_preds = list(model.predict(X.iloc[train]))
            train_true = list(y.iloc[train])

            test_preds_0 = list(model.predict(X.iloc[test[0]]))
            test_preds_1 = list(model.predict(X.iloc[test[1]]))
            test_true_0 = list(y.iloc[test[0]])
            test_true_1 = list(y.iloc[test[1]])

            backtest_paths[a][0] = backtest_paths[a][0] + test_preds_0
            backtest_paths[a][1] = backtest_paths[a][1] + test_true_0
            backtest_paths[b][0] = backtest_paths[b][0] + test_preds_1
            backtest_paths[b][1] = backtest_paths[b][1] + test_true_1

            if count == path_tracker:
                a_start += 1
                a = a_start
                count = 0
                b += 1
                path_tracker -= 1
            else:
                a += 1
                count += 1

        if return_raw:
            for i in range(n_backtest_paths):
                preds = preds + backtest_paths[i][0]
                true_vals = true_vals + backtest_paths[i][1]
                try: pred_probas = pred_probas + list(model.predict_proba(backtest_paths[i][0])[:, 1])
                except: pass
            raw_output = {
                'preds': preds,
                'true_vals': true_vals,
                'probas': pred_probas,
                'residuals': residuals,
            }
            return raw_output
        else:
            scorer = scorer._score_func # scoring function must be a partial function or else the kwargs won't be passed
            train_scores.append(scorer(train_true, train_preds))
            for i in range(n_backtest_paths):
                test_scores.append(scorer(backtest_paths[i][1], backtest_paths[i][0]))
            return train_scores, test_scores


class CombPurgedCV():
    """ Combinatorial Purged Cross-Validation (CPCV):

        :param N: (int) total number of folds
        :param k: (int) number of test folds
        :param purge: (float) number of observations to purge
        :param embargo: (float) number of observations to embargo
    """
    def __init__(self, N=6, k=2, purge=0, embargo=0):
        self.N = N
        self.k = k
        self.purge = purge
        self.embargo = embargo

    def split(self, X, y):
        """ Split the data into train and test sets. Yield train/test indices for each split.

            :param X: (pd.DataFrame) features
            :param y: (pd.Series) target
            :yield: (tuple) train/test indices
        """
        groups = np.array_split(X.index.to_numpy(), self.N)
        for test_groups in combinations(np.arange(self.N), self.k):
            test_indices = {group: groups[group] for group in test_groups}
            train_indices = np.concatenate([groups[group] for group in range(self.N) if group not in test_groups])
            for group in test_indices.values():
                 train_indices = train_indices[np.logical_or(train_indices > group[0], train_indices < group[0] - self.purge)]
                 train_indices = train_indices[np.logical_or(train_indices < group[-1], train_indices > group[-1] + self.embargo)]
            yield train_indices, tuple(test_indices.values())

    def get_n_splits(self):
        """ Get the number of train/test splits
        """
        prod = 1
        for i in range(0, self.k):
            prod *= (self.N - i)
        return int(prod / math.factorial(self.k))

    def get_n_backtest_paths(self):
        """ Get the number of independent backtest paths generated
              n_backtest_paths = n_splits * (k / N)
        """
        prod = 1
        for i in range(1, self.k):
            prod *= (self.N - i)
        return int(prod / math.factorial(self.k - 1))

    def __str__(self):
        return f'CombPurgedCV(N={self.N}, k={self.k}, purge={self.purge}, embargo={self.embargo})'
