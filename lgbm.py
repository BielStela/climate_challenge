# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 11:21:39 2019

@author: borre001
"""

import csv
from hyperopt import STATUS_OK
from timeit import default_timer as timer
import numpy as np
from hyperopt import hp
from hyperopt import tpe, Trials, fmin
import lightgbm as lgb
from utils_train import load_data, load_results, save_results
from functools import partial
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from time import sleep
from argparse import ArgumentParser

N_FOLDS = 3
MAX_EVALS = 500
ITERATION = 0

SPACE = {
    'boosting_type': hp.choice('boosting_type',
                               [{'boosting_type': 'gbdt',
                                 'subsample': hp.uniform('gdbt_subsample',
                                                         0.5, 1)},
                                {'boosting_type': 'dart',
                                 'subsample': hp.uniform('dart_subsample',
                                                         0.5, 1)},
                                {'boosting_type': 'goss',
                                 'subsample': 1.0}]),
    'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
    'learning_rate': hp.loguniform('learning_rate',
                                   np.log(0.01), np.log(0.2)),
    'n_estimators': hp.quniform('n_estimators', 100, 500, 25),
    'subsample_for_bin': hp.quniform('subsample_for_bin',
                                     20000, 300000, 20000),
    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
}


def parse():

    parser = ArgumentParser()
    parser.add_argument("--mode", dest="mode", help="default or optimize",
                        type=str, default="default")
    parsed = parser.parse_args()
    return parsed


def objective(params, X, y, out_file):
    """Objective function for Gradient Boosting
    Machine Hyperparameter Optimization"""

    # Keep track of evals
    global ITERATION

    ITERATION += 1

    # Retrieve the subsample if present otherwise set to 1.0
    subsample = params['boosting_type'].get('subsample', 1.0)

    # Extract the boosting type
    params['boosting_type'] = params['boosting_type']['boosting_type']
    params['subsample'] = subsample

    # Make sure parameters that need to be integers are integers
    for parameter_name in ['num_leaves',
                           'subsample_for_bin',
                           'min_child_samples']:
        params[parameter_name] = int(params[parameter_name])
    params['objective'] = 'regression'
    params['n_jobs'] = -1
    params['metric'] = 'mse'
    params['verbose'] = 0
    # params['device_type'] = 'gpu'

    start = timer()

    # Perform n_folds cross validation
    results = []
    fold = KFold(n_splits=N_FOLDS, shuffle=True)

    for train_idx, test_idx in fold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_set = lgb.Dataset(X_train, label=y_train)
        test_set = lgb.Dataset(X_test, label=y_test)
        bst = lgb.train(params, train_set,
                        early_stopping_rounds=100,
                        valid_sets=[test_set], verbose_eval=0)
        y_pred = bst.predict(X_test)
        results.append(mean_squared_error(y_test, y_pred))

    run_time = timer() - start

    # Extract the best score
    best_score = np.mean(results)

    # Write to the csv file ('a' means append)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([best_score, params, ITERATION, run_time])

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


def lgbm_default(X, y, df, variables, i, name):

    if (df['name'] == name).sum():
        print("Already computed")
        return df, i
    else:
        params = {"learning_rate": 0.1,
                  "num_leaves": 255,
                  "num_trees": 500,
                  "min_data_in_leaf": 0,
                  "min_sum_hessian_in_leaf": 100}
        params['objective'] = 'regression'
        params['n_jobs'] = -1
        params['metric'] = 'mse'
        params['verbose'] = 0
        results = []
        fold = KFold(n_splits=10, shuffle=True)

        for train_idx, test_idx in fold.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            train_set = lgb.Dataset(X_train, label=y_train)
            test_set = lgb.Dataset(X_test, label=y_test)
            bst = lgb.train(params, train_set,
                            valid_sets=[test_set], verbose_eval=0)
            y_pred = bst.predict(X_test)
            results.append(mean_squared_error(y_test, y_pred))

        df = save_results(df, name, np.mean(results),
                          variables, i)

        return df, i + 1


if __name__ == "__main__":
    X, y, variables,X_test, days = load_data()
    print("Data Loaded")
    df, i = load_results()

    parsed = parse()
    if parsed.mode == "default":
        df, i = lgbm_default(X,
                             y,
                             df,
                             variables,
                             i,
                             "lgbm_default_full_vars")
    else:
        optimize(X, y, objective, name='lgbm_trials.csv', space=SPACE,
                 max_evals=MAX_EVALS)
