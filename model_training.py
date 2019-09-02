#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 20:46:43 2019

@author: juan

File to train the models.
"""

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from pandas import read_csv, DataFrame
from math import sqrt
import numpy as np
from os import path


def load_data():

    scaler = StandardScaler()

    X = read_csv("./data_for_models/X.csv").values
    y = read_csv("./data_for_models/y.csv")['T_MEAN'].values

    scaler.fit(X)
    X_std = scaler.transform(X)

    return X_std, y, X.columns


def save_results(file, name, score, variables, index):

    file[index] = [name, score, variables]
    file.to_csv("results.csv")
    return file


def load_results(name="./results.csv"):
    if path.exists(name):
        df = read_csv(name)
        index = len(df)
        return df, index
    else:
        df = DataFrame(columns=['name', 'score', 'variables'])
        return df, 0


def random_forest_default(X, y, df, variables, i):

    results = []
    fold = KFold(n_splits=5, shuffle=True)

    for train_idx, test_idx in fold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        forest = RandomForestRegressor(n_jobs=-1)
        forest.fit(X_train, y_train)
        y_pred = forest.predict(X_test)

        results.append(sqrt(mean_squared_error(y_test, y_pred)))

    df = save_results(df, "random_forest_default", np.mean(results), variables,
                      i)

    return df, i + 1


if __name__ == "__main__":

    X, y, variables = load_data()
    df, i = load_results()

    df, i = random_forest_default(X, y, df,  variables, i)
