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
from utils_train import load_data, load_results
from functools import partial
from tqdm import tqdm


N_FOLDS = 5
MAX_EVALS = 3

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


def objective(params, train_set, out_file, n_folds, pbar):
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

    start = timer()

    # Perform n_folds cross validation
    cv_results = lgb.cv(params, train_set, num_boost_round=10000,
                        nfold=n_folds, early_stopping_rounds=100,
                        metrics='mse', seed=42)

    run_time = timer() - start

    # Extract the best score
    best_score = np.max(cv_results['mse-mean'])

    # Boosting rounds that returned the highest cv score
    n_estimators = int(np.argmax(cv_results['mse-mean']) + 1)

    # Write to the csv file ('a' means append)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([best_score, params, ITERATION, n_estimators, run_time])

    pbar.update()
    # Dictionary with information for evaluation
    return {'loss': best_score, 'params': params, 'iteration': ITERATION,
            'estimators': n_estimators,
            'train_time': run_time, 'status': STATUS_OK}


def main():

    X, y, variables = load_data()
    print("Data Loaded")
    df, i = load_results()

    bayes_trials = Trials()

    out_file = 'gbm_trials.csv'
    of_connection = open(out_file, 'w')
    writer = csv.writer(of_connection)

    # Write the headers to the file
    writer.writerow(['loss', 'params', 'iteration', 'estimators',
                     'train_time'])
    of_connection.close()
    train_set = lgb.Dataset(X, label=y)

    global ITERATION

    ITERATION = 0
    pbar = tqdm(total=MAX_EVALS, desc="Hyperopt")
    objective_partial = partial(objective, train_set=train_set,
                                out_file=out_file,
                                n_folds=N_FOLDS, pbar=pbar)

    # Run optimization
    fmin(fn=objective_partial, space=SPACE, algo=tpe.suggest,
         max_evals=MAX_EVALS, trials=bayes_trials,
         rstate=np.random.RandomState(np.random.randint(0, 100, 1)))

    pbar.close()
    bayes_trials_results = sorted(bayes_trials.results,
                                  key=lambda x: x['loss'])
    print(bayes_trials_results[:2])


if __name__ == "__main__":
    main()