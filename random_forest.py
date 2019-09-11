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
from hyperopt import hp, STATUS_OK, Trials, fmin, tpe
from timeit import default_timer as timer
import numpy as np
import csv
from hyperopt.pyll.base import scope
from time import sleep
from functools import partial

N_FOLDS = 3
MAX_EVALS = 100
ITERATION = 0

SPACE = {
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 200, 10)),
        # criterion
        'max_depth': scope.int(hp.quniform('max_depth', 2, 22, 4)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2,
                                                   102, 10)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf',
                                                  1, 102, 10)),
        # min_weight_fraction_leaf
        # 'max_features'
        # max_leaf_nodes
        # min_impurity_decrease
        'bootstrap': hp.choice('bootstrap', [{'bootstrap': True,
                                              'oob_score': True},
                                             {'bootstrap': False,
                                              'oob_score': False}])
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
        fold = KFold(n_splits=10, shuffle=True)

        for train_idx, test_idx in fold.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            forest = RandomForestRegressor(n_jobs=-1)
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

    params['oob_score'] = params['bootstrap']['oob_score']
    params['bootstrap'] = params['bootstrap']['bootstrap']

    for j, (train_idx, test_idx) in enumerate(fold.split(X)):
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


def optimize(X, y, objective, name, space, max_evals):

    bayes_trials = Trials()

    out_file = name
    of_connection = open(out_file, 'w')
    writer = csv.writer(of_connection)

    # Write the headers to the file
    writer.writerow(['loss', 'params', 'iteration',
                     'train_time'])
    of_connection.close()

    objective_partial = partial(objective, X=X, y=y,
                                out_file=out_file)

    # Run optimization
    fmin(fn=objective_partial, space=space, algo=tpe.suggest,
         max_evals=max_evals, trials=bayes_trials,
         rstate=np.random.RandomState(np.random.randint(0, 100, 1)))

    bayes_trials_results = sorted(bayes_trials.results,
                                  key=lambda x: x['loss'])
    print(bayes_trials_results[:2])


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
                                      "random_forest_default_normal_vars")

    elif parsed.mode == "optimize":
        optimize(X, y, objective, name='forest_trials_full_vars.csv',
                 space=SPACE, max_evals=MAX_EVALS)

    elif parsed.mode == "predict":
        y_pred = predict_with_forest(X_test, X, y)
        give_pred_format(X_test, y_pred, "random_forest.csv", days)