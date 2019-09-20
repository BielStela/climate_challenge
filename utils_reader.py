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
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA


np.random.seed(42)
test_size = 0.2

OFFICIAL_ATTR = [['DATA', 'Tm', 'Tx', 'Tn', 'ESTACIO', 'PPT24h', 'HRm',
                  'hPa', 'RS24h', 'VVem6', 'DVum6', 'VVx6', 'DVx6'],
                 ['DATA', 'Tm', 'Tx', 'Tn', 'ESTACIO', 'HRm'],
                 ['DATA', 'Tm', 'Tx', 'Tn', 'ESTACIO', 'PPT24h', 'HRm',
                  'hPa', 'RS24h', 'VVem10', 'DVum10', 'VVx10', 'DVx10'],
                 ['DATA', 'Tm', 'Tx', 'Tn', 'ESTACIO', 'PPT24h', 'HRm',
                  'hPa', 'RS24h', 'VVem10', 'DVum10', 'VVx10', 'DVx10']]

OFFICIAL_ATTR_HOURLY = [['DATA', 'T', 'TX', 'TN', 'HR', 'HRN',
                         'HRX', 'PPT', 'VVM10', 'DVM10', 'VVX10', 'DVVX10',
                         'Unnamed: 13'],
                        ['DATA', 'T', 'Tx', 'Tn', 'HR', 'HRN', 'HRX'],
                        ['DATA', 'T', 'Tx', 'Tn', 'HR', 'HRn', 'HRx', 'PPT',
                         'VVM10', 'DVM10', 'VVX10', 'DVX10', 'RS'],
                        ['DATA', 'T', 'Tx', 'Tn', 'HR', 'HRn', 'HRx', 'PPT',
                         'VVM10', 'DVM10', 'VVX10', 'DVX10', 'RS']
                        ]

UNOFFICIAL_ATTR = [['DATA', 'Alt', 'Temp_Max', 'Temp_Min', 'Hum_Max',
                    'Hum_Min', 'Pres_Max', 'Pres_Min', 'Wind_Max']]


def parse_arguments(parser):
    parser.add_argument("-d", dest="comp_dist",
                        help="compute distances and save them in a file",
                        type=int,
                        default=0)
    parser.add_argument("-s", dest="save",
                        help="save x and y in a folder ./data_for_models",
                        type=int,
                        default=1)
    parser.add_argument("-n", dest="no_official",
                        help="compute qwith no official",
                        type=int,
                        default=1)
    parser.add_argument("-p", dest="pred",
                        help="save for rpediction",
                        type=int,
                        default=0)

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
        return X.loc[:, self.attribute_names]


class official_station_adder(BaseEstimator, TransformerMixin):

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
        self.full = X
        jj = len(self.full.columns)
        for i, j in tqdm(zip(self.attributes_to_add,
                             range(len(self.attributes_to_add))),
                         total=len(self.attributes_to_add)):
            y[j]['DATA'] = pd.to_datetime(y[j]['DATA'],
                                          format="%Y-%m-%d", exact=True)
            y[j]['DATA'] = y[j]['DATA'].dt.strftime("%Y-%m-%d")
            self.full = pd.merge(self.full, y[j][i], how='inner',
                                 left_on='day', right_on='DATA',
                                 suffixes=("_" + str(jj), "_" +
                                           str(jj+1)))
            jj += 1

        if self.include_distance:
            self.full = pd.merge(self.full, self.distances, how='inner',
                                 left_on='idx', right_on='idx',
                                 suffixes=("_0", "_1"))

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

    full_file['day'] = pd.to_datetime(full_file['day'], format="%d/%m/%Y",
                                      exact=True)
    full_file['day'] = full_file['day'].dt.strftime("%Y-%m-%d")
    return full_file


def read_official_stations(name="./climateChallengeData/data_S2_S3_S4.xlsx"):
    return pd.read_excel(name, delimeter=";", sheet_name=[0, 1, 2, 3])


def read_hourly_official(direc="./climateChallengeData/data_hours"):

    hour_files = []
    for i in listdir(direc):
        file = pd.read_csv(path.join(direc, i), delimiter=";")
        file['date'] = pd.to_datetime(file['DATA'],
                                      format="%d/%m/%Y %H:%M",
                                      errors='coerce').dt.date
        mask = file.date.isnull()
        file.loc[mask, 'date'] = pd.to_datetime(file['DATA'],
                                                format="%d/%m/%Y",
                                                errors='coerce').dt.date
        file['DATA'] = pd.to_datetime(file['date'],
                                      format="%Y-%m-%d",
                                      exact=True)
        file['year'] = file['DATA'].dt.year
        file['day'] = file['DATA'].dt.day
        file_filtered = file[file.year.isin(['2012', '2013',
                                             '2014', '2015',
                                             '2016'])]
        file_filtered = file_filtered[file_filtered.day.isin(['1',
                                                              '2',
                                                              '3',
                                                              '4',
                                                              '5'])]
        hours_grouped = file_filtered.groupby('DATA',
                                              as_index=False).agg(
                                                      pd.Series.mean)
        hour_files.append(hours_grouped)

    return hour_files


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


def save_data_folder(X, y=None, direc="./data_for_models/",
                     name_X="X.parquet.gzip",
                     name_y=None):

    if not path.exists(direc):
        mkdir(direc)

    X.to_parquet(path.join(direc, name_X), compression='gzip')
    if name_y is not None:
        y.to_parquet(path.join(direc, name_y), compression='gzip')


def read_unofficial_data(day=6):
    unofficial = pd.read_excel("./climateChallengeData/data_NoOfficial.xlsx",
                               delimiter=";")
    unofficial['Date'] = pd.to_datetime(unofficial['Date'], format="%d/%m/%Y",
                                        exact=True)
    unofficial['DATA'] = unofficial['Date'].dt.strftime("%Y-%m-%d")
    unofficial['ndays_year'] = unofficial['Date'].dt.day
    unofficial = unofficial[unofficial['ndays_year'] < day]
    unofficial.drop(columns=['Date', 'ndays_year'], inplace=True)

    return unofficial


def give_basic_dataset(include_distance=0, official_attr=None):
    full_real = read_real_files()
    official_stations_daily = read_official_stations()

    if include_distance:
        official_stations_latlon = pd.read_csv("./climateChallengeData/officialStations.csv")
        grid_points = pd.read_csv("./climateChallengeData/grid2latlon.csv")
        compute_distances(official_stations_latlon, grid_points)
        create_idx(full_real)
    else:
        grid_points = None

    df_full = official_station_adder(
        official_attr,
        include_distance=include_distance,
        distances=grid_points).transform(
        full_real, official_stations_daily)

    df_full.drop(columns=['nx', 'ny', 'LAT', 'LON', 'T_MIN', 'T_MAX'] +
                         [i for i in df_full.columns if 'ESTACIO_' in i] +
                         [i for i in df_full.columns if 'DATA_' in i],
                 inplace=True)
    return df_full


def input_(X):
    imputer = SimpleImputer(strategy='median')
    X_t = imputer.fit_transform(X.values)
    return X_t


def give_n_add_hourly(prev_data=None, official_attr_hourly=None):
    official_stations_hourly = read_hourly_official()
    df_full = official_station_adder(official_attr_hourly,
                                     include_distance=False,
                                     distances=None).transform(
                                             prev_data,
                                             official_stations_hourly)
    return df_full


def result_pca(df, threshold=0.97):
    full_a = df.loc[:, ['day', 'T_MEAN']]
    X = df.loc[:,
               df.columns[(df.columns != 'T_MEAN') & (df.columns != 'day')]]
    X_t = input_(X)
    pca = PCA(n_components=threshold)
    X_n = pca.fit_transform(X_t)
    columns = ["Feature_" + str(i) for i in range(len(X.columns),
                                                  len(X.columns) +
                                                  len(X_n[0, :]))]
    new = pd.DataFrame(data=X_n, columns=columns)
    full = pd.concat([full_a, new], ignore_index=False, axis=1)
    return full


def give_n_add_nonoff_dataset(include_distance=0, prev_data=None,
                              nonofficial_attr=None):
    if include_distance:
        non_official_stations_latlon = pd.read_csv("./climateChallengeData/noOfficialStations.csv")
        grid_points = pd.read_csv("./climateChallengeData/grid2latlon.csv")
        compute_distances(non_official_stations_latlon,
                          grid_points,
                          file_name="./climateChallengeData/distances_noofficial.csv")
    else:
        grid_points_non_official = None
        unofficial = read_unofficial_data()
        df_full = official_station_adder(
                nonofficial_attr,
                include_distance=include_distance,
                distances=grid_points_non_official).transform(
                        prev_data, [unofficial])
    
    df_full.drop(columns=['DATA'])
    return df_full


def prepare_data(include_distance=0, save_data=1,
                 add_not_official=False, add_hourly_data=False,
                 official_attr_hourly=OFFICIAL_ATTR_HOURLY,
                 nonofficial_attr=UNOFFICIAL_ATTR,
                 official_attr=OFFICIAL_ATTR,
                 with_pca=True):

    download_files()
    full = give_basic_dataset(include_distance=include_distance,
                              official_attr=official_attr)

    full = result_pca(full)

    if add_not_official:
        full = give_n_add_nonoff_dataset(include_distance=include_distance,
                                         nonofficial_attr=nonofficial_attr,
                                         prev_data=full)
        if with_pca:
            full = result_pca(full)
            
    print(full.info())

    if add_hourly_data:
        full = give_n_add_hourly(prev_data=full,
                                 official_attr_hourly=official_attr_hourly)
        print(full.info())
        if with_pca:
            full = result_pca(full)
        

    y_columns = ['T_MEAN']
    x_columns = full.columns[full.columns != 'T_MEAN']

    X = DataFrameSelector(x_columns).transform(full)
    y = DataFrameSelector(y_columns).transform(full)

    X.drop(columns=['day'],
           inplace=True)
    if include_distance:
        X.drop(columns=['idx'], inplace=True)

    if save_data:
        save_data_folder(X, y, name_y="y.parquet.gzip")
        full.to_parquet("./data_for_models/full.parquet.gzip",
                        compression='gzip')
    else:
        return X, y


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parsed = parse_arguments(parser)

    prepare_data(include_distance=parsed.comp_dist,
                 save_data=parsed.save,
                 add_not_official=parsed.no_official,
                 add_hourly_data=False,
                 with_pca=True)
#    file_for_prediction_n_submission()