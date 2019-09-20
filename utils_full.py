# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 20:47:56 2019

@author: borre001
"""

from utils_reader import read_real_files, read_official_stations
from utils_reader import read_unofficial_data, read_hourly_official
from utils_reader import official_station_adder,  result_pca
from utils_reader import OFFICIAL_ATTR, UNOFFICIAL_ATTR, OFFICIAL_ATTR_HOURLY
import pandas as pd
import numpy as np
from lgbm import predict_with_lgbm
from utils_train import give_pred_format


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
        unofficial = read_unofficial_data()
        features = ['Wind_Max', 'Temp_Max', 'Temp_Min']
        unofficial[features] = unofficial[features].apply(
                lambda x: pd.to_numeric(x))
        the_other = [unofficial]
    else:
        the_other = read_hourly_official()

    df = official_station_adder(attr,
                                include_distance=None,
                                distances=None).transform(entry,
                                                          the_other)
    return df


def first_frame(mode="train", with_pca=True):

    if mode == "train":
        real = full_real()
    else:
        real = dates_f_prediction()

    df = add_df(real, OFFICIAL_ATTR)
    df = add_df(df, UNOFFICIAL_ATTR, "unofficial")
    df = add_df(df, OFFICIAL_ATTR_HOURLY, "hourly")

    drop_function(df, "ESTACIO_")
    drop_function(df, "DATA")

    if with_pca:
        df = result_pca(df)

    if mode == "train":
        y = df['T_MEAN']
        X = df.loc[:, df.columns[(df.columns != 'T_MEAN') &
                                 (df.columns != 'day')]]
        return X, y
    else:
        return df[df.columns[df.columns != 'day']], None, df['day']

def second_frame(mode='train'):
    
    if mode == "train":
        real = full_real(full=1)
    else:
        real = dates_f_prediction(full=1)

        
if __name__ == "__main__":

    X, y = first_frame()
    X_pred, _, days = first_frame(mode="predict")
    y_pred = predict_with_lgbm(X_pred, X, y)
    give_pred_format(X_pred, y_pred, "./submission/lgbm.csv", days)
    
    