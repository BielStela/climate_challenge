# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 20:47:56 2019

@author: borre001
"""

from utils_reader import read_real_files, read_official_stations
from os import path, mkdir
from utils_reader import read_unofficial_data, read_hourly_official
from utils_reader import official_station_adder,  result_pca, input_scale
import pandas as pd
import numpy as np
from utils_train import give_pred_format
from bestmodel import default_model_predict
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# BEST: lasso with T and day alpha =0.1
# correlation threshold +- 0.3

#OFFICIAL_ATTR = [['DATA', 'Tm', 'ESTACIO'],
#                 ['DATA', 'Tm', 'ESTACIO'],
#                 ['DATA', 'Tm',
#                  'ESTACIO'],
#                 ['DATA', 'Tm', 
#                  'ESTACIO']]

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
#    sample = ['Alt', 'Temp_Max', 'Temp_Min', 'Hum_Max',
#              'Hum_Min', 'Pres_Max', 'Pres_Min', 'Wind_Max',
#              'Prec_Today',
#              'Prec_Year']
    sample = ['Temp_Max', 'Temp_Min']
    final_labels = []
    for i in sample:
        for j in nonoff:
            final_labels.append(str(i) + str(j))
    return UNOFFICIAL_ATTR + final_labels


def full_real(full=0):
    real = read_real_files()

    if not full:
        real = real.loc[:, ['day', 'T_MEAN']]
        real = real.groupby("day", as_index=False).agg({"T_MEAN": "mean"})
    else:
        real = real.loc[:, ['day', 'T_MEAN', 'LAT', 'LON']]
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
                                          'LAT': grid_points['LAT'].values,
                                          'LON': grid_points['LON'].values})
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
                pca=None, threshold=0.8, features=None, corr=True,
                kbest=None, selector=None):

    if mode == "train":
        real = full_real()
    else:
        real = dates_f_prediction()

    df = add_df(real, OFFICIAL_ATTR)
    df['ndays'] = pd.to_datetime(df['day'], format="%Y-%m-%d",
                                 exact=True).dt.dayofyear
    # df = add_df(df, OFFICIAL_ATTR_HOURLY, "hourly")
    # df = add_df(df, [unofficial_attr()], "unofficial")

    drop_function(df, "ESTACIO_")
    drop_function(df, "DATA")

    if mode == "train":
        if corr:
            corr = df.corr()['T_MEAN']
            features = [j for i, j in zip(corr, corr.index) if i > 0.8]
            df = df.loc[:, features + ['day']]
        else:
            features = None
        if with_pca:
            df, pca, imputer, scaler = result_pca(df, threshold=threshold)
            y = df['T_MEAN']
            X = df.loc[:, df.columns[(df.columns != 'T_MEAN') &
                                     (df.columns != 'day')]]
        else:
            y = df['T_MEAN']
            X = df.loc[:, df.columns[(df.columns != 'T_MEAN') &
                                     (df.columns != 'day')]]
            X, imputer, scaler = input_scale(X)
            pca = None
        if kbest:
            selector = SelectKBest(f_regression, k=5)
            X = selector.fit_transform(X, y)
        else:
            selector = None
        return X, y, pca, imputer, scaler, features, selector, df['day']
    else:
        if corr:
            df = df.loc[:, features + ['day']]
        if with_pca:
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
        if kbest:
            X = selector.transform(X)
        return X, None, df['day']


def generate_2frame():
    range_dates = pd.date_range(start='1/1/2016', end='31/12/2016', freq="D")
    gridtolatlon = pd.read_csv("./climateChallengeData/grid2latlon.csv")
    number_points = len(gridtolatlon)
    final_df = pd.DataFrame(columns=['day', 'LAT', 'LON'])
    for i in range_dates:
        day = pd.to_datetime(i, format="%Y-%m-%d")
        if day.day > 5:
            day = day.strftime("%Y-%m-%d")

            days = np.repeat(day, number_points)
            df = pd.DataFrame({'day': days, 'LAT': gridtolatlon['LAT'],
                               'LON': gridtolatlon['LON']})
            final_df = pd.concat([final_df, df], ignore_index=True)
    return final_df


def second_frame(model2=Ridge(alpha=0.1), mode='train',
                 features=['green_co_3', 'slope_mean'],
                 scaler=StandardScaler(),
                 imputer=SimpleImputer()):

    if mode == 'train':
        real = full_real(full=1)
    else:
        real = generate_2frame()

    extra_data = pd.read_csv("extra_features.csv")
    gridtolatlon = pd.read_csv("./climateChallengeData/grid2latlon.csv")
    extra_data['LAT'], extra_data['LON'] = gridtolatlon['LAT'], gridtolatlon['LON']
    extra_data.drop(columns=extra_data.columns[(extra_data.columns != 'LON') &
                                               (extra_data.columns != 'LAT') &
                                               (extra_data.columns != features[0]) &
                                               (extra_data.columns != features[1])
                                               ], inplace=True)
    real_full = pd.merge(real, extra_data, on=['LAT', 'LON'])

    real = add_df(real_full, OFFICIAL_ATTR)
    drop_function(real, "ESTACIO_")
    drop_function(real, "DATA")
    real['ndays'] = pd.to_datetime(real['day'], format="%Y-%m-%d",
                                 exact=True).dt.dayofyear
    if mode == 'train':
        imputed = imputer.fit_transform(real.loc[:, real.columns[
                (real.columns != 'day') & (real.columns != 'LAT') &
                (real.columns != 'LON') & (real.columns != 'T_MEAN')]])
        reescaled = scaler.fit_transform(imputed)

        model2.fit(reescaled, real['T_MEAN'])
        y_pred = None
        days = None
    else:
        imputed = imputer.transform(real.loc[:, real.columns[
                (real.columns != 'day') & (real.columns != 'LAT') &
                (real.columns != 'LON') & (real.columns != 'T_MEAN')]])
        reescaled = scaler.transform(imputed)

        y_end = model2.predict(reescaled)
        frame_agg = pd.DataFrame({'day': real['day'], 'LAT': real['LAT'],
                                  'LON': real['LON'], 'y_pred': y_end})
        end_frame = frame_agg.groupby(by='day').agg({'y_pred': 'mean'})
        y_pred = end_frame['y_pred'].values
        days = end_frame.index
    return model2, imputer, scaler, y_pred, days


def save_data_numpy(X, y=None, direc="./data_for_models/",
                    name_X="X.npy",
                    name_y=None):

    if not path.exists(direc):
        mkdir(direc)

    np.save(path.join(direc, name_X), X)
    if name_y is not None:
        np.save(path.join(direc, name_y), y)


if __name__ == "__main__":
    threshold = 0.97
    # retorna X=X_train, y=y_train
    X, y, pca, imputer, scaler, features, selector, days = first_frame(
            threshold=threshold,
            corr=False, kbest=False)
    X_pred, _, days = first_frame(mode="predict", imputer=imputer,
                                  scaler=scaler, pca=pca,
                                  threshold=threshold, features=features,
                                  corr=False, kbest=False, selector=selector)
    save_data_numpy(X, y, name_y="y.npy")
    save_data_numpy(X_pred, name_X="X_test.npy")
    y_pred = default_model_predict(Ridge(alpha=0.1), X, y, X_pred)
#    model2, imputer_2, scaler_2, pred_2, days = second_frame()
#    model2, imputer_2, scaler_2, y_pred, days = second_frame(model2=model2,
#                                                             mode='test',
#                                                             scaler=scaler_2,
#                                                             imputer=imputer_2
#                                                             )


    give_pred_format(X_pred, y_pred, "./submission/linear_2frame.csv", days)
