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
from pandas import read_csv
from math import sqrt


def load_data():
    scaler = StandardScaler()

    X = read_csv("./data_for_models/X.csv").values
    y = read_csv("./data_for_models/y.csv")['T_MEAN'].values

    scaler.fit(X)
    X_std = scaler.transform(X)

    return X_std, y


if __name__ == "__main__":
    load_data()
    
    fold = KFold(n_splits=5, shuffle=True)

    for train_idx, test_idx in fold.split(X_std):
        X_train, X_test = X_std[train_idx], X_std[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        forest = RandomForestRegressor(n_jobs=-1)
        forest.fit(X_train, y_train)
        y_pred = forest.predict(X_test)

        print(sqrt(mean_squared_error(y_test, y_pred)))
