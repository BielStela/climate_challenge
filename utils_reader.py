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

import geopy.distance as distance
from tqdm import tqdm as tqdm

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


OFFICIAL_ATTR = [['DATA', 'Tm', 'Tx', 'Tn', 'PPT24h', 'HRm',
                  'hPa', 'RS24h', 'VVem6', 'DVum6', 'VVx6', 'DVx6', 'ESTACIO'],
                 ['DATA', 'Tm', 'Tx', 'Tn', 'HRm', 'ESTACIO'],
                 ['DATA', 'Tm', 'Tx', 'Tn', 'PPT24h', 'HRm',
                  'hPa', 'RS24h', 'VVem10', 'DVum10', 'VVx10', 'DVx10',
                  'ESTACIO'],
                 ['DATA', 'Tm', 'Tx', 'Tn', 'PPT24h', 'HRm',
                  'hPa', 'RS24h', 'VVem10', 'DVum10', 'VVx10', 'DVx10',
                  'ESTACIO']]

OFFICIAL_ATTR_HOURLY = [['DATA', 'T', 'TX', 'TN', 'HR', 'HRN', 'HRX', 'PPT',
                         'VVM10', 'DVM10', 'VVX10', 'DVVX10', 'Unnamed: 13'],
                        ['DATA',  'T', 'Tx', 'Tn', 'HR', 'HRN', 'HRX'],
                        ['DATA', 'T', 'Tx', 'Tn', 'HR', 'HRn', 'HRx', 'PPT',
                         'VVM10', 'DVM10', 'VVX10', 'DVX10', 'RS'],
                        ['DATA',  'T', 'Tx', 'Tn', 'HR', 'HRn', 'HRx', 'PPT',
                         'VVM10', 'DVM10', 'VVX10', 'DVX10', 'RS']
                        ]

UNOFFICIAL_ATTR = [['DATA', 'Alt', 'Temp_Max', 'Temp_Min', 'Hum_Max',
                    'Hum_Min', 'Pres_Max', 'Pres_Min', 'Wind_Max',
                    'Prec_Today',
                    'Prec_Year']]


def create_partial():

    real_2016 = pd.read_csv("./climateChallengeData/real/real_2016.csv",
                            index_col=None)
    real_2016 = real_2016.groupby('day',
                                  as_index=False).agg({'T_MEAN': 'mean'})
    real_2016.columns = ['date', 'mean']
    real_2016.to_csv("./data_for_models/sub_partial.csv")


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
                                 suffixes=("_" + str(jj),
                                           "_" + str(jj+1)))
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
    files = [i for i in listdir(direc)]
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    for i in files:
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
        hours_grouped = file_filtered.groupby('DATA',
                                              as_index=False).agg(
                                                      pd.Series.mean)
        file['DATA'] = pd.to_datetime(file['DATA'],
                                      format="%Y-%m-%d",
                                      exact=True)
        file['DATA'] = file['DATA'].dt.strftime("%Y-%m-%d")
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
                     name_X="X.csv",
                     name_y=None):

    if not path.exists(direc):
        mkdir(direc)

    X.to_csv(path.join(direc, name_X))
    if name_y is not None:
        y.to_csv(path.join(direc, name_y))


def read_unofficial_data(day=6):
    unofficial = pd.read_excel("./climateChallengeData/data_NoOfficial.xlsx",
                               delimiter=";", index_col=0)
    unofficial['Date'] = pd.to_datetime(unofficial['Date'], format="%d/%m/%Y",
                                        exact=True)
    unofficial['DATA'] = unofficial['Date'].dt.strftime("%Y-%m-%d")
    unofficial['ndays_year'] = unofficial['Date'].dt.day
    unofficial = unofficial[unofficial['ndays_year'] < day]
    unofficial.drop(columns=['Date', 'ndays_year'], inplace=True)
    return unofficial


def input_scale(X):
    imputer = SimpleImputer(strategy='median')
    X_t = imputer.fit_transform(X.values)
    scaler = StandardScaler()
    X_t = scaler.fit_transform(X_t)
    return X_t, imputer, scaler


def result_pca(df, threshold=0.95, scaler=None, imputer=None, pca=None):
    full_a = df.loc[:, ['day', 'T_MEAN']]
    X = df.loc[:,
               df.columns[(df.columns != 'T_MEAN') & (df.columns != 'day')]]
    if pca is not None:
        X_t = imputer.transform(X)
        X_t = scaler.transform(X_t)
        X_n = pca.transform(X_t)
    else:
        X_t, imputer, scaler = input_scale(X)
        pca = PCA(n_components=threshold)
        X_n = pca.fit_transform(X_t)
    columns = ["Feature_" + str(i) for i in range(len(full_a.columns),
                                                  len(full_a.columns) +
                                                  len(X_n[0, :]))]
    new = pd.DataFrame(data=X_n, columns=columns)
    full = pd.concat([full_a, new], ignore_index=False, axis=1)
    return full, pca, imputer, scaler
