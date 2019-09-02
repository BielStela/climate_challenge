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
from utils_train import save_results, load_results, load_data


def random_forest_default(X, y, df, variables, i):

    results = []
    fold = KFold(n_splits=5, shuffle=True)

    for train_idx, test_idx in fold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        forest = RandomForestRegressor(n_jobs=-1)
        forest.fit(X_train, y_train)
        y_pred = forest.predict(X_test)

        results.append(mean_squared_error(y_test, y_pred))

    df = save_results(df, "random_forest_default_full_vars", mean(results),
                      variables, i)

    return df, i + 1


if __name__ == "__main__":

    X, y, variables = load_data()
    df, i = load_results()

    df, i = random_forest_default(X, y, df,  variables, i)
