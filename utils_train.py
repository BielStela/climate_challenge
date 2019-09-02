#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 09:57:09 2019

@author: juan
"""

from os import path
from pandas import read_csv, DataFrame
from sklearn.preprocessing import StandardScaler, Imputer


def load_data():

    scaler = StandardScaler()

    X = read_csv("./data_for_models/X.csv")
    columns = X.columns
    y = read_csv("./data_for_models/y.csv")['T_MEAN']

    imputer = Imputer(strategy='median')
    X = imputer.fit_transform(X.values)

    scaler.fit(X)
    X_std = scaler.transform(X)

    return X_std, y.values, columns


def save_results(file, name, score, variables, index):

    file.loc[index] = [name, score, variables]
    file.to_csv("./results.csv")
    return file


def load_results(name="./results.csv"):
    if path.exists(name):
        df = read_csv(name, index_col=0)
        index = len(df)
        return df, index
    else:
        df = DataFrame(columns=['name', 'score', 'variables'])
        return df, 0
