# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 18:49:11 2019

@author: borre001
"""
from utils_reader import OFFICIAL_ATTR, OFFICIAL_ATTR_HOURLY, UNOFFICIAL_ATTR
from utils_reader import save_data_folder, compute_distances
from utils_reader import official_station_adder
from utils_reader import give_n_add_nonoff_dataset, give_n_add_hourly
from utils_reader import read_official_stations, create_idx
import pandas as pd
import numpy as np


def prepare_file_sub():
    real_2016 = pd.read_csv("./climateChallengeData/real/real_2016.csv",
                            index_col=None)
    real_2016 = real_2016.groupby('day',
                                  as_index=False).agg({'T_MEAN': 'mean'})
    real_2016.columns = ['date', 'mean']
    real_2016.to_pickle("./data_for_models/sub_partial.pkl")
    grid_points = pd.read_csv("./climateChallengeData/grid2latlon.csv")

    final_df = []
    data_ranges = pd.date_range(start='01/01/2016', end='31/12/2016')
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
    final_df['day'] = pd.to_datetime(final_df['day'],
                                     format="%d/%m/%Y", exact=True)
    final_df['day'] = final_df['day'].dt.strftime("%Y-%m-%d")
    return final_df, grid_points


def add_official(prev_data, include_distance=None, official_attr=None,
                 grid=None):
    if include_distance:
        official_stations_latlon = pd.read_csv("./climateChallengeData/officialStations.csv")
        create_idx(prev_data)
        compute_distances(official_stations_latlon, grid)

    official_stations_daily = read_official_stations()

    adder = official_station_adder(official_attr,
                                   include_distance=False,
                                   distances=grid)
    X_complete = adder.transform(
        prev_data, official_stations_daily)

    X_complete.drop(columns=['nx', 'ny'] +
                            [i for i in X_complete.columns if 'ESTACIO_' in i] + 
                            [i for i in X_complete.columns if 'DATA_' in i],
                    inplace=True)
    return X_complete


def file_for_prediction_n_submission(include_distance=False,
                                     official_attr=OFFICIAL_ATTR,
                                     add_hourly_data=False,
                                     add_nonofficial=False,
                                     nonofficial_attr=UNOFFICIAL_ATTR,
                                     official_attr_hourly=OFFICIAL_ATTR_HOURLY):

    df, grid = prepare_file_sub()

    full = add_official(df, include_distance=include_distance,
                        official_attr=official_attr, grid=grid)

    if add_nonofficial:
        full = give_n_add_nonoff_dataset(include_distance=include_distance,
                                         nonofficial_attr=nonofficial_attr,
                                         prev_data=full)
    if add_hourly_data:
        full = give_n_add_hourly(prev_data=full,
                                 official_attr_hourly=official_attr_hourly)

    save_data_folder(full, name_X="for_submission.pkl")