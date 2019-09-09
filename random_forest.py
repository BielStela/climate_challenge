#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 20:46:43 2019

@author: juan

File to train the models.
"""

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from numpy import mean
from utils_train import save_results, load_results, load_data, give_pred_format
from argparse import ArgumentParser
from hyperopt import hp, STATUS_OK
from timeit import default_timer as timer
import numpy as np
import csv
from lgbm import optimize
from hyperopt.pyll.base import scope
from time import sleep
import pandas as pd

N_FOLDS = 3
MAX_EVALS = 100
ITERATION = 0

SPACE = {
        'n_estimators': scope.int(hp.quniform('n_estimators', 20, 500, 20)),
        # criterion
        'max_depth': scope.int(hp.quniform('max_depth', 2, 20, 2)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2,
                                                   100, 5)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf',
                                                  1, 10, 2))
        # min_weight_fraction_leaf
        # 'max_features'
        # max_leaf_nodes
        # min_impurity_decrease
#        'bootstrap': hp.choice('bootstrap', [{'bootstrap': True,
#                                              'oob_score': True},
#                                             {'bootstrap': False,
#                                              'oob_score': False}])
        }


def parse():

    parser = ArgumentParser()
    parser.add_argument("--mode", dest="mode", help="default or optimize",
                        type=str, default="default")
    parsed = parser.parse_args()
    return parsed


def random_forest_default(X, y, df, variables, i, name):

    if (df['name'] == name).sum():
        print("Already computed")
        return df, i
    else:
        results = []
        fold = KFold(n_splits=5, shuffle=True)

        for train_idx, test_idx in fold.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            forest = RandomForestRegressor(n_jobs=-1,
                                           n_estimators=300)
            forest.fit(X_train, y_train)
            y_pred = forest.predict(X_test)

            results.append(mean_squared_error(y_test, y_pred))

        df = save_results(df, name, mean(results),
                          variables, i)

        return df, i + 1


def predict_with_forest(X_test, X, y):

    
    forest = RandomForestRegressor(n_jobs=-1)
    forest.fit(X, y)
    y_pred = forest.predict(X_test)
    
    return y_pred



def objective(params, X, y, out_file):
    """Objective function for Gradient Boosting
    Machine Hyperparameter Optimization"""

    # Keep track of evals
    global ITERATION

    ITERATION += 1

    start = timer()

    # Perform n_folds cross validation
    results = []
    fold = KFold(n_splits=N_FOLDS, shuffle=True)

#    params['oob_score'] = params['bootstrap']['oob_score']
#    params['bootstrap'] = params['bootstrap']['bootstrap']
    print(params['n_estimators'], params['max_depth'])

    for j, (train_idx, test_idx) in enumerate(fold.split(X)):
        print(j)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        forest = RandomForestRegressor(n_jobs=-1,
                                       random_state=np.random.randint(42, 100))
        forest.set_params(**params)
        forest.fit(X_train, y_train)
        y_pred = forest.predict(X_test)
        results.append(mean_squared_error(y_test, y_pred))

    run_time = timer() - start

    # Extract the best score
    best_score = np.mean(results)

    # Write to the csv file ('a' means append)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([best_score, params, ITERATION, run_time])
    sleep(120)
    # Dictionary with information for evaluation
    return {'loss': best_score, 'params': params, 'iteration': ITERATION,
            'train_time': run_time, 'status': STATUS_OK}


if __name__ == "__main__":

    X, y, variables, X_test, days = load_data()
    df, i = load_results()

    parsed = parse()
    if parsed.mode == "default":
        df, i = random_forest_default(X,
                                      y,
                                      df,
                                      variables,
                                      i,
                                      "random_forest_default_300_estim_normal_vars")

    elif parsed.mode == "optimize":
        optimize(X, y, objective, name='forest_trials_full_vars.csv',
                 space=SPACE, max_evals=MAX_EVALS)

    elif parsed.mode == "predict":
        y_pred = predict_with_forest(X_test, X, y)
        give_pred_format(X_test, y_pred, "random_forest.csv", days)