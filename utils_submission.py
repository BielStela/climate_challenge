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


def create_partial():

    real_2016 = pd.read_csv("./climateChallengeData/real/real_2016.csv",
                            index_col=None)
    real_2016 = real_2016.groupby('day',
                                  as_index=False).agg({'T_MEAN': 'mean'})
    real_2016.columns = ['date', 'mean']
    real_2016.to_csv("./data_for_models/sub_partial.csv")


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

