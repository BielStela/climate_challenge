#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 20:46:43 2019

@author: juan

File to train the models.
"""

from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from pandas import read_csv

test_size = 0.2


def main():
    X = read_csv("./data_for_models/X.csv")
    y = read_csv("./data_for_models/y.csv")

    fold = KFold(n_splits=10, shuffle=True)

    for train_idx, test_idx in fold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        svr = SVR.fit(X_train, y_train)
        y_pred = svr.predict(X_test)

        print(accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    main()
