#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 09:57:09 2019

@author: juan
"""

from os import path
from pandas import read_csv, DataFrame
from sklearn.preprocessing import StandardScaler, Imputer
import pandas as pd

def give_pred_format(X_test, y_pred, name, days):
    df = pd.DataFrame(data={'date': days, 'mean': y_pred}, index=None)
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d", exact=True)
    df = df.groupby('date', as_index=False).agg({'mean': 'mean'})
    df['date'] = df['date'].dt.strftime("%d/%m/%Y")
    df_partial = pd.read_csv("./data_for_models/sub_partial.csv", index_col=0)
    df_partial['date'] = pd.to_datetime(df_partial['date'], format="%d/%m/%Y",
                                        exact=True)
    df_partial['date'] = df_partial['date'].dt.strftime("%d/%m/%Y")
    result = pd.concat([df, df_partial], ignore_index=True)
    result['date'] = pd.to_datetime(result['date'], format="%d/%m/%Y",
                                    exact=True)
    result.sort_values('date', inplace=True)
    result.reset_index(inplace=True, drop=True)
    result['date'] = result['date'].dt.strftime("%d/%m/%Y")
    result.to_csv(name, columns=['date', 'mean'], index=False)


def load_data(name_x="./data_for_models/X.csv",
              name_y="./data_for_models/y.csv",
              name_test="./data_for_models/for_submission.csv"):

    scaler = StandardScaler()

    X = read_csv(name_x)
    columns = X.columns
    y = read_csv(name_y)['T_MEAN']

    imputer = Imputer(strategy='median')
    X = imputer.fit_transform(X.values)

    scaler.fit(X)
    X_std = scaler.transform(X)

    X_test = read_csv(name_test)
    X_test_pred = X_test.loc[:, X_test.columns != 'day'].values
    days = X_test['day']

    X_test_pred = imputer.transform(X_test_pred)
    X_test_pred = scaler.transform(X_test_pred)

    return X_std, y.values, columns, X_test_pred, days


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
