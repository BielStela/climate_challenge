# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 20:47:56 2019

@author: borre001
"""

from utils_reader import read_real_files, read_official_stations
from utils_reader import read_unofficial_data, read_hourly_official
from utils_reader import official_station_adder,  result_pca, input_scale
import pandas as pd
import numpy as np
from lgbm import predict_with_lgbm
from random_forest import predict_with_forest
from utils_train import give_pred_format

# correlation threshold +- 0.3

OFFICIAL_ATTR = [['DATA', 'Tm', 'ESTACIO'],
                 ['DATA', 'Tm', 'ESTACIO'],
                 ['DATA', 'Tm',
                  'ESTACIO'],
                 ['DATA', 'Tm', 
                  'ESTACIO']]

OFFICIAL_ATTR_HOURLY = [['DATA', 'T', 'TX', 'TN', 'HR', 'HRN', 'HRX', 'PPT',
                         'VVM10', 'DVM10', 'VVX10', 'DVVX10', 'Unnamed: 13'],
                        ['DATA',  'T', 'Tx', 'Tn', 'HR', 'HRN', 'HRX'],
                        ['DATA', 'T', 'Tx', 'Tn', 'HR', 'HRn', 'HRx', 'PPT',
                         'VVM10', 'DVM10', 'VVX10', 'DVX10', 'RS'],
                        ['DATA',  'T', 'Tx', 'Tn', 'HR', 'HRn', 'HRx', 'PPT',
                         'VVM10', 'DVM10', 'VVX10', 'DVX10', 'RS']
                        ]


def unofficial_attr():
    nonoff = np.unique(read_unofficial_data(day=32)['Station'])
    UNOFFICIAL_ATTR = ['DATA']
    sample = ['Alt', 'Temp_Max', 'Temp_Min', 'Hum_Max',
              'Hum_Min', 'Pres_Max', 'Pres_Min', 'Wind_Max',
              'Prec_Today',
              'Prec_Year']
    final_labels = []
    for i in sample:
        for j in nonoff:
            final_labels.append(str(i) + str(j))
    return UNOFFICIAL_ATTR + final_labels

def full_real(full=0):
    real = read_real_files()
    real = real.loc[:, ['day', 'T_MEAN']]
    if not full:
        real = real.groupby("day", as_index=False).agg({"T_MEAN": "mean"})
    return real


def drop_function(df, string):
    df.drop(columns=[i for i in df.columns if string in i],
            inplace=True)


def dates_f_prediction(full=0):
    final_df = []
    data_ranges = pd.date_range(start='01/01/2016', end='31/12/2016')
    grid_points = pd.read_csv("./climateChallengeData/grid2latlon.csv")
    if full:
        data_days = data_ranges.dayofyear - 1
        for i, j in zip(data_ranges, data_days):
            if i.day > 5:
                dates = np.repeat(i, len(grid_points))
                days = np.repeat(j, len(grid_points))
                sample_df = pd.DataFrame({'day': dates, 'ndays': days,
                                          'nx': grid_points['nx'].values,
                                          'ny': grid_points['ny'].values})
                final_df.append(sample_df)
        final_df = pd.concat(final_df, ignore_index=True, sort=False)
    else:
        final_df = pd.DataFrame({'day': data_ranges})
    final_df['day'] = pd.to_datetime(final_df['day'],
                                     format="%d/%m/%Y", exact=True)
    final_df['day'] = final_df['day'].dt.strftime("%Y-%m-%d")
    return final_df


def add_df(entry, attr, other="official"):

    if other == "official":
        the_other = read_official_stations()
    elif other == "unofficial":
        unofficial = read_unofficial_data(day=32)
        features = ['Wind_Max', 'Temp_Max', 'Temp_Min']
        unofficial[features] = unofficial[features].apply(
                lambda x: pd.to_numeric(x))
        unofficial = unofficial.pivot(index='DATA',
                                      columns='Station').reset_index()
        unofficial.columns = [str(i) + str(j) for i, j in unofficial.columns]
        the_other = [unofficial]
    else:
        the_other = read_hourly_official()

    adder = official_station_adder(attr, include_distance=None,
                                   distances=None)
    df = adder.transform(entry, the_other)
    return df


def first_frame(mode="train", with_pca=False, imputer=None, scaler=None,
                pca=None, threshold=0.8, features = None):

    if mode == "train":
        real = full_real()
    else:
        real = dates_f_prediction()

    df = add_df(real, OFFICIAL_ATTR)
    # df = add_df(df, OFFICIAL_ATTR_HOURLY, "hourly")
    # df = add_df(df, [unofficial_attr()], "unofficial")

    drop_function(df, "ESTACIO_")
    drop_function(df, "DATA")

    if mode == "train":
        corr = df.corr()['T_MEAN']
        features = [j for i, j in zip(corr, corr.index) if i > 0.3]
        df = df.loc[:, features + ['day']]
        if with_pca:
            df, pca, imputer, scaler = result_pca(df, threshold=threshold)
        else:
            y = df['T_MEAN']
            X = df.loc[:, df.columns[(df.columns != 'T_MEAN') &
                                     (df.columns != 'day')]]
            X, imputer, scaler = input_scale(X)
            features, pca = None, None
        return X, y, pca, imputer, scaler, features
    else:
        
        if with_pca:
            df = df.loc[:, features + ['day']]
            df, pca, imputer, scaler = result_pca(df, scaler=scaler,
                                                  imputer=imputer, pca=pca,
                                                  threshold=threshold)
            df['days'] = pd.to_datetime(df['day'],
                                        format="%Y-%m-%d", exact=True).dt.day
            df = df[df['days'] > 5]
            X = df.loc[:, df.columns[(df.columns != 'T_MEAN') &
                                     (df.columns != 'day') &
                                     (df.columns != 'days')]]
        else:
            df['days'] = pd.to_datetime(df['day'],
                                        format="%Y-%m-%d", exact=True).dt.day
            df = df[df['days'] > 5]
            X = df.loc[:, df.columns[(df.columns != 'T_MEAN') &
                                     (df.columns != 'day') &
                                     (df.columns != 'days')]]
            X = imputer.transform(X)
            X = scaler.transform(X)
        return X, None, df['day']


def second_frame(mode='train'):

    if mode == "train":
        real = full_real(full=1)
    else:
        real = dates_f_prediction(full=1)


if __name__ == "__main__":
    threshold = 0.97
    # retorna X=X_train, y=y_train
    X, y, pca, imputer, scaler, features = first_frame(threshold=threshold)
    #aquest es Xpred
    X_pred, _, days = first_frame(mode="predict", imputer=imputer,
                                  scaler=scaler, pca=pca,
                                  threshold=threshold, features=features)
    # prediu ambn lgbm
    y_pred = predict_with_forest(X_pred, X, y)
    #dona la predicció amb format d'entrega
    give_pred_format(X_pred, y_pred, "./submission/lgbm.csv", days)
    
    