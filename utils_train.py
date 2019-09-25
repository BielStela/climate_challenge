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
from utils_reader import official_station_adder
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
import pickle
from time import sleep
from utils_reader import create_partial
from tqdm import tqdm


OFFICIAL_ATTR_2 = [['DATA', 'Tm'],
                   ['DATA', 'Tm'],
                   ['DATA', 'Tm'],
                   ['DATA', 'Tm']]

ROOT = "./submission/results_task_2/"


def give_pred_format(X_test, y_pred, name, days):
    df = pd.DataFrame(data={'date': days, 'mean': y_pred}, index=None)
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d", exact=True)
    df = df.groupby('date', as_index=False).agg({'mean': 'mean'})
    df['date'] = df['date'].dt.strftime("%d/%m/%Y")
    if not path.exists("./data_for_models/sub_partial.csv"):
        create_partial()
    df_partial = pd.read_csv("./data_for_models/sub_partial.csv")
    df_partial['date'] = pd.to_datetime(df_partial['date'], format="%d/%m/%Y",
                                        exact=True)
    df_partial['date'] = df_partial['date'].dt.strftime("%d/%m/%Y")
    result = pd.concat([df, df_partial], ignore_index=True, sort=True)
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

    X_t = X[X.columns[(X.columns != 'T_MEAN') &
                      (X.columns != 'idx') & (X.columns != 'day')]].values
    y = X['T_MEAN'].values

    results = []
    fold = KFold(n_splits=5, shuffle=True)

    for train_idx, test_idx in fold.split(X_t):
        X_train, X_test = X_t[train_idx], X_t[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # linear = LinearRegression(n_jobs=-1)
        linear = Ridge(alpha=0.1)
        linear.fit(X_train, y_train)
        y_pred = linear.predict(X_test)

        results.append(mean_squared_error(y_test, y_pred))

    # linear = LinearRegression(n_jobs=-1)
    linear = Ridge(alpha=0.1)
    linear.fit(X_t, y)

    return results, linear.coef_, linear.intercept_


def give_n_points_n_weights(df, n_points=1, identifier='idx', method="mean"):

    # 0 for values of mse out of bag, 1 for coefs and 2 for intercept
    list_mins = [[], [], []]

    list_ids = np.unique(df[identifier])

    for i in tqdm(list_ids):

        X = df[df[identifier] == i]
        errors, coefs, intercepts = classify_one_idx(X)
        if method == "mean":
            values = np.mean(errors)
        elif method == "std":
            values = np.std(errors)
        for j, h in enumerate(list([values, coefs, intercepts])):
            list_mins[j].append(h)

        new_points = list_ids[np.argpartition(list_mins[0],
                                              -n_points)[-n_points:]]

    return new_points, list_mins[1], list_mins[2], list_ids, list_mins[0]


def train_2nd_task_min_error(include_distance=False,
                             official_attr=OFFICIAL_ATTR_2,
                             method="mean"):

    full_real = read_real_files()
    official_stations_daily = read_official_stations()
    official_stations_latlon = pd.read_csv("./climateChallengeData/officialStations.csv")
    create_idx(full_real)
    full_real.drop(columns=['ndays', 'T_MIN', 'T_MAX',
                            'LAT', 'LON', 'nx', 'ny'], inplace=True)

    df_full = official_station_adder(
        official_attr,
        include_distance=include_distance).transform(
        full_real, official_stations_daily)

    df_full.drop(columns=[i for i in
                          df_full.columns if 'DATA' in i], inplace=True)

    new_points, coefs, intercepts, list_ids, errors = give_n_points_n_weights(df_full,
                                                                              method=method)

    full_coef_list = []
    full_intercept_list = []
    full_id_list = []
    points = []
    error_list = []

    points.append(new_points)

    for count in tqdm(range(1, 11)):
        extra = []
        for i in new_points:
            extra.append(full_real[full_real['idx'] == i])
            # full_real = full_real[full_real['idx'] != i]

        for j, i in enumerate(extra):
            df_full = df_full.merge(i, how='inner',
                                    left_on='day', right_on='day',
                                    suffixes=("", "_" +
                                              str(count)), copy=False)
        df_full.drop(columns=['idx_' + str(i) for i in
                              range(count, count+1)], inplace=True)

        new_points, coefs, intercepts, list_ids, errors = give_n_points_n_weights(df_full,
                                                                                  method=method)
        points.append(new_points)
        error_list.append(errors)
        if count in list([1, 2, 5, 10]):
            full_coef_list.append(coefs)
            full_intercept_list.append(intercepts)
            full_id_list.append(list_ids)

        sleep(60)

    dump_pickle(path.join(ROOT, "coefs.pkl"), full_coef_list)
    dump_pickle(path.join(ROOT, "intercepts.pkl"), full_intercept_list)
    dump_pickle(path.join(ROOT, "list_ids.pkl"), full_id_list)
    dump_pickle(path.join(ROOT, "points.pkl"), points)
    dump_pickle(path.join(ROOT, "error.pkl"), error_list)


def dump_pickle(name, arr):
    with open(name, 'wb') as fp:
        pickle.dump(arr, fp)


def read_pickle(name):
    with open(name, 'rb') as fp:
        return pickle.load(fp)


def give_prediction_2nd_task():
    coefs = read_pickle(path.join(ROOT, "coefs.pkl"))
    inter = read_pickle(path.join(ROOT, "intercepts.pkl"))
    ids = read_pickle(path.join(ROOT, "list_ids.pkl"))
    points = read_pickle(path.join(ROOT, "points.pkl"))

    index = ['e5', 'e5', 'e6', 'e5', 'e6', 'e7', 'e8', 'e9',
             'e5', 'e6', 'e7', 'e8', 'e9', 'e10', 'e11', 'e12', 'e13', 'e14']
    positions = [[], []]

    for i, j in enumerate([1, 1, 2, 1, 2, 3, 4, 5,
                           1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
        positions[0].append(int(points[j][0][:3]))
        positions[1].append(int(points[j][0][3:]))

    df1 = pd.DataFrame({'est': index, 'nx': positions[0], 'ny': positions[1]})

    dump_pickle(path.join(ROOT, "1st_dataframe.pkl"), df1)

    columns = ['nx', 'ny', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8',
               'w9', 'w10', 'w11', 'w12', 'w13', 'w14', 'b']
    df2 = pd.DataFrame(columns=columns)

    for k, i in tqdm(enumerate(ids[0])):
        for j in range(len(ids)):
            length = len(coefs[j][0])
            new_dict = {}
            new_dict['nx'] = int(i[:3])
            new_dict['ny'] = int(i[3:])
            for h in range(length):
                new_dict[columns[h+2]] = coefs[j][k][h]
            for h in range(length+2, len(columns)-1):
                new_dict[columns[h]] = "-"
            new_dict['b'] = inter[j][k]
            df2 = df2.append(new_dict, ignore_index=True)

    dump_pickle(path.join(ROOT, "2nd_dataframe.pkl"), df2)


def arrange_pred_2ndtask():
    part1 = read_pickle(path.join(ROOT, "1st_dataframe.pkl"))
    part2 = read_pickle(path.join(ROOT, "2nd_dataframe.pkl"))

    part1.to_csv(path.join(ROOT, "prediction.csv"), index=False)
    part2['nx'] = part2['nx'].astype(int)
    part2['ny'] = part2['ny'].astype(int)
    part2.to_csv(path.join(ROOT, "prediction.csv"), mode='a', index=False)


if __name__ == "__main__":
    train_2nd_task_min_error()
