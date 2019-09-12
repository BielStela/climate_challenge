#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 22:33:01 2019

@author: BCJuan

Functions for the pipeline devoted to create the full dataset

All the files (.csv) must be in a folder named data. The structure is the same
as the zip dowloaded from http://climate-challenge.herokuapp.com/data/.
However there is a new folder named examples which has the sample
submission files.

All the data is saved in data (if stated so in args).
So the file should only be called once.

However you can call the file from another file without saving

"""

from sklearn.base import BaseEstimator, TransformerMixin
from os import path, listdir, mkdir
import pandas as pd
import urllib.request as req
from zipfile import ZipFile
from io import BytesIO
import numpy as np
import geopy.distance as distance
from tqdm import tqdm as tqdm
import argparse

np.random.seed(42)
test_size = 0.2

OFFICIAL_ATTR = [['DATA', 'Tm', 'Tx', 'Tn', 'ESTACIO', 'PPT24h', 'HRm',
                  'hPa', 'RS24h', 'VVem6', 'DVum6', 'VVx6', 'DVx6'],
                 ['DATA', 'Tm', 'Tx', 'Tn', 'ESTACIO', 'HRm'],
                 ['DATA', 'Tm', 'Tx', 'Tn', 'ESTACIO', 'PPT24h', 'HRm',
                  'hPa', 'RS24h', 'VVem10', 'DVum10', 'VVx10', 'DVx10'],
                 ['DATA', 'Tm', 'Tx', 'Tn', 'ESTACIO', 'PPT24h', 'HRm',
                  'hPa', 'RS24h', 'VVem10', 'DVum10', 'VVx10', 'DVx10']]
#
#OFFICIAL_ATTR = [['DATA', 'Tm', 'Tx', 'Tn', 'ESTACIO', 'HRm', 'RS24h'],
#                 ['DATA', 'Tm', 'Tx', 'Tn', 'ESTACIO', 'HRm'],
#                 ['DATA', 'Tm', 'Tx', 'Tn', 'ESTACIO', 'HRm', 'RS24h'],
#                 ['DATA', 'Tm', 'Tx', 'Tn', 'ESTACIO', 'HRm', 'RS24h']]


def parse_arguments(parser):
    parser.add_argument("-d", dest="comp_dist",
                        help="compute distances and save them in a file",
                        type=int,
                        default=1)
    parser.add_argument("-s", dest="save",
                        help="save x and y in a folder ./data_for_models",
                        type=int,
                        default=1)
    parser.add_argument("-n", dest="no_official",
                        help="compute qwith no official",
                        type=int,
                        default=0)
    parser.add_argument("-p", dest="pred",
                        help="save for rpediction",
                        type=int,
                        default=1)

    return parser.parse_args()


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Class for selecting columns from a dataframe by using a list
    with the column names
    """
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.loc[:,self.attribute_names]


class official_station_daily_adder(BaseEstimator, TransformerMixin):

    """
    Class for adding attributes from official stations to the values predicted
    from the model (real values). The data is the joint for the moment
    """

    def __init__(self, attributes_to_add,
                 include_distance=True, distances=None):
        self.attributes_to_add = attributes_to_add
        self.include_distance = include_distance

        if self.include_distance:
            self.distances = distances

    def fit(self, X, y=None):
        return self

    def transform(self, X, y):
        # to_datetime functions is super slow if format is not supplied
        X['day'] = pd.to_datetime(X['day'], infer_datetime_format=True)
        self.full = X.copy()

        for i, j in tqdm(zip(self.attributes_to_add,
                             range(len(self.attributes_to_add))),
                         total=len(self.attributes_to_add)):
            self.full = self.full.merge(y[j][i], how='inner',
                                        left_on='day', right_on='DATA',
                                        suffixes=("_" + str(j-1), "_" +
                                                  str(j)))

        if self.include_distance:
            create_idx(self.full)
            self.full = self.full.merge(self.distances, how='inner',
                                        left_on='idx', right_on='idx')

        return self.full


def read_real_files(direc="./climateChallengeData/real"):
    """
    Takes the files in the folder real and groups them in a sole pandas
    THis dataframe will be the base for adding more features and make the
    training val and test division

    args
    -----
    direc (numeric):
        where the files are located

    returns
    ------

    full_file (dataframe):
        dataframe with all the years and each day withe corresponding
        data
    """
    files = []
    for i in listdir(direc):
        name = path.join(direc, i)
        files.append(pd.read_csv(name))
    full_file = pd.concat(files)
    return full_file


def read_official_stations(name="./climateChallengeData/data_S2_S3_S4.xlsx"):
    return pd.read_excel(name, delimeter=";", sheet_name=[0, 1, 2, 3])


def compute_distances(latLonStations, gridPoints,
                      file_name="./climateChallengeData/distances.csv"):

    for i in range(len(latLonStations)):
        gridPoints['dist'+str(i)] = gridPoints.apply(
                lambda x: distance.geodesic(
                        (x['LAT'], x['LON']),
                        (latLonStations.loc[i, 'Latitude'],
                         latLonStations.loc[i, 'Longitude'])).km,
                axis=1)
    create_idx(gridPoints)
    gridPoints.drop(columns=['nx', 'ny', 'LAT', 'LON'], inplace=True)
    gridPoints.to_csv(file_name)

def create_idx(df):
    df['idx'] = df['nx'].astype(str) + df['ny'].astype(str)


def download_files(direc="./climateChallengeData/"):
    """
    Downloads files and puts them in a common folder. Dowloads main zip and
    sample submissions
    """

    def give_request(url):
        """
        Returns request content of url
        """
        request = req.Request(url)
        ff = req.urlopen(request).read()
        return ff

    def save_sample(content, name, direc, folder_name="sample_submissions"):
        """
        Saves the sample submissions file inside a scpecific folder in the data
        folder
        """
        if not path.exists(path.join(direc, folder_name)):
            mkdir(path.join(direc, folder_name))
        with open(path.join(direc, folder_name, name), 'wb') as f:
            f.write(content)

    if not path.exists(direc):
        url = "http://climate-challenge.herokuapp.com/climateApp/static/data/climateChallengeData.zip"
        url_sample1 = "http://climate-challenge.herokuapp.com/climateApp/static/sub_example/S1.csv"
        url_sample2 = "http://climate-challenge.herokuapp.com/climateApp/static/sub_example/S2.csv"

        print("Downloading \n", end="\r")

        with ZipFile(BytesIO(give_request(url)), 'r') as z:

            print("Extracting data \n", end="\r")
            mkdir(direc)
            z.extractall(path=direc)

        save_sample(give_request(url_sample1), "S1.csv", direc)
        save_sample(give_request(url_sample2), "S2.csv", direc)

    else:
        print("Data already downloaded")


def save_data_folder(X, y=None, direc="./data_for_models/", name_X="X.csv",
                     name_y=None):

    if not path.exists(direc):
        mkdir(direc)

    X.to_csv(path.join(direc, name_X))
    if name_y is not None:
        y.to_csv(path.join(direc, name_y))


def prepare_data(include_distance=1, save_data=1, official_attr=OFFICIAL_ATTR,
                 add_not_official=False):

    download_files()
    full_real = read_real_files()
    official_stations_daily = read_official_stations()
    official_stations_latlon = pd.read_csv("./climateChallengeData/officialStations.csv")
    grid_points = pd.read_csv("./climateChallengeData/grid2latlon.csv")

    compute_distances(official_stations_latlon, grid_points)
    create_idx(full_real)

    df_full = official_station_daily_adder(
        official_attr,
        include_distance=include_distance, distances=grid_points).transform(
        full_real, official_stations_daily)

    y_columns = ['T_MEAN']
    x_columns = df_full.columns[df_full.columns != 'T_MEAN']

    X = DataFrameSelector(x_columns).transform(df_full)
    y = DataFrameSelector(y_columns).transform(df_full)

    X.drop(columns=['day', 'nx', 'ny', 'idx', 'LAT', 'LON', 'T_MIN', 'T_MAX'] +
                   ['DATA_' + str(i) for i in
                    range(len(official_stations_latlon))] +
                   ['ESTACIO_' + str(i) for i in
                    range(len(official_stations_latlon))], inplace=True)

    if add_not_official:
        unofficial_stations_latlon = pd.read_csv("./climateChallengeData/officialStations.csv")

    if save_data:
        save_data_folder(X, y, name_y="y.csv")
    else:
        return X, y


def file_for_prediction_n_submission(include_distance=True,
                                     official_attr=OFFICIAL_ATTR):

    real_2016 = pd.read_csv("./climateChallengeData/real/real_2016.csv",
                            index_col=None)
    real_2016 = real_2016.groupby('day', as_index=False).agg({'T_MEAN': 'mean'})
    real_2016.columns = ['date', 'mean']
    real_2016.to_csv("./data_for_models/sub_partial.csv")

    grid_points = pd.read_csv("./climateChallengeData/grid2latlon.csv")
    official_stations_latlon = pd.read_csv("./climateChallengeData/officialStations.csv")

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

    create_idx(final_df)
    compute_distances(official_stations_latlon, grid_points)

    official_stations_daily = read_official_stations()

    X_complete = official_station_daily_adder(
        official_attr,
        include_distance=include_distance, distances=grid_points).transform(
        final_df, official_stations_daily)

    X_complete.drop(columns=['nx', 'ny', 'idx'] +
                    ['DATA_' + str(i) for i in
                     range(len(official_stations_latlon))] +
                    ['ESTACIO_' + str(i) for i in
                     range(len(official_stations_latlon))], inplace=True)

    save_data_folder(X_complete, name_X="for_submission.csv")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parsed = parse_arguments(parser)

    prepare_data(include_distance=parsed.comp_dist,
                 save_data=parsed.save)
    file_for_prediction_n_submission()