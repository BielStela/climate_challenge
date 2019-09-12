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
from utils_reader import read_real_files, read_official_stations, create_idx
from utils_reader import official_station_daily_adder
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm 

OFFICIAL_ATTR_2 = [['DATA', 'Tm'],
                   ['DATA', 'Tm'],
                   ['DATA', 'Tm'],
                   ['DATA', 'Tm']]

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


def classify_one_idx(X):

    scaler = StandardScaler()
    X_t = scaler.fit_transform(
            X[X.columns[(X.columns != 'T_MEAN') &
                        (X.columns != 'idx') & (X.columns != 'day')]])
    y = X['T_MEAN'].values

    results = []
    fold = KFold(n_splits=10, shuffle=True)

    for train_idx, test_idx in fold.split(X_t):
        X_train, X_test = X_t[train_idx], X_t[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        linear = LinearRegression(n_jobs=-1)
        linear.fit(X_train, y_train)
        y_pred = linear.predict(X_test)

        results.append(mean_squared_error(y_test, y_pred))

    linear = LinearRegression()
    linear.fit(X_t, y)

    return np.mean(results), linear.coef_, linear.intercept_


def give_n_points_n_weights(df, n_points=1, identifier='idx'):

    # 0 for values of mse out of bag, 1 for coefs and 2 for intercept
    list_mins = [[], [], []]
    for i in tqdm(np.unique(df[identifier])):
        X = df[df[identifier] == i]
        m, c, i = classify_one_idx(X)
        for j, i in enumerate(list([m, c, i])):
            list_mins[j].append(i)

    new_points = np.unique(df[identifier])[
            np.argpartition(a, -n_points)[-n_points:]]
    return new_points, list_mins[0], list_mins[1]


def train_2nd_task_min_error(include_distance=False,
                   official_attr=OFFICIAL_ATTR_2):

    full_real = read_real_files()
    official_stations_daily = read_official_stations()
    official_stations_latlon = pd.read_csv("./climateChallengeData/officialStations.csv")
    create_idx(full_real)
    full_real.drop(columns=['ndays', 'T_MIN', 'T_MAX',
                            'LAT', 'LON', 'nx', 'ny'], inplace=True)

    df_full = official_station_daily_adder(
        official_attr,
        include_distance=include_distance).transform(
        full_real, official_stations_daily)

    df_full.drop(columns=['DATA_' + str(i) for i in 
                          range(len(official_stations_latlon))], inplace=True)

    new_points, coefs, intercepts = give_n_points_n_weights(df_full)
    
    extra = []
    for (i, j) in points:
        query = str(i) + str(j)
        extra.append(full_real[full_real['idx'] == query])
        full_real = full_real[full_real['idx'] != query]
        
    for j, i in enumerate(extra):
        i['day'] = pd.to_datetime(i['day'], infer_datetime_format=True)
        df_full = df_full.merge(i, how='inner',
                            left_on='day', right_on='day',
                            suffixes=("_" + str(j), "_" +
                                      str(j+1)))

    df_full.drop(columns=['idx_' + str(i) for i in 
                          range(1, len(extra)+1)] + ['day'], inplace=True)