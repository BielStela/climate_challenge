#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 09:43:46 2019

@author: cephis
"""

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error

import numpy as np
import pandas as pd
import sklearn.gaussian_process as gp

def default_model_test(model, X_train, y_train, X_test, y_test):
    mod = model
    mod.fit(X_train, y_train)
    y_pred = mod.predict(X_test)
    return mean_squared_error(y_test, y_pred), np.sum(np.abs(y_test-y_pred))

def default_model_predict(model, X_train, y_train, X_test):
    mod = model
    mod.fit(X_train, y_train)
    y_pred = mod.predict(X_test)
    return y_pred

def choose_best_default_model(X, y):
    
    models = [RandomForestRegressor(n_jobs=-1),
              LinearRegression(n_jobs=-1),
              Ridge(alpha=0.1),
              Lasso(alpha=0.1),
              xgboost.XGBRegressor(objective='reg:linear'),
              AdaBoostRegressor(base_estimator=Ridge(alpha=0.1),
                                learning_rate=0.01, n_estimators=100,
                                loss='linear'),
              AdaBoostRegressor(base_estimator=DecisionTreeRegressor(),
                                learning_rate=0.5),
              SVR(),
              ]
    results = []
    fold = KFold(n_splits=10, shuffle=True)
    
    for i, j in enumerate(models):
        absolute = []
        for train_idx, test_idx in fold.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
        
            s, a = default_model_test(j, X_train, y_train, X_test, y_test)
            
            absolute.append(a)
        
        results.append([np.sum(absolute)])

    return results
    
    
def load_data_numpy(name_x="./data_for_models/X.npy",
              name_y="./data_for_models/y.npy",
              name_test="./data_for_models/X_test.npy"):
    X = np.load(name_x)
    y = np.load(name_y)
    X_test = np.load(name_test)
    return X, y, X_test

if __name__ == "__main__":
    X, y, X_test = load_data_numpy()
    results = choose_best_default_model(X, y)
    print(results)
    