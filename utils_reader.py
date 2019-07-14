#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 22:33:01 2019

@author: BCJuan

Functions for the pipeline devoted to create training, validation and test

All the files (.csv) must be in a folder named data. The structure is the same
as the zip dowloaded from http://climate-challenge.herokuapp.com/data/.
However there is a new folder named examples which has the sample
submission files
"""

from sklearn.base import BaseEstimator, TransformerMixin
from os import path, listdir, mkdir
import pandas as pd
import urllib.request as req
from zipfile import ZipFile
from io import BytesIO
from sklearn.model_selection import train_test_split
import numpy as np
import geopy.distance as distance
from tqdm import tqdm as tqdm
import argparse

np.random.seed(42)
test_size = 0.2


def parse_arguments(parser):
    parser.add_argument("-d", dest="comp_dist",
                        help="compute distances and save them in a file",
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
        return X[self.attribute_names]


class official_station_daily_adder(BaseEstimator, TransformerMixin):

    """
    Class for adding attributes from official stations to the values predicted
    from the model (real values). The data is the joint for the moment
    """

    def __init__(self, attributes_to_add, station_locations,
                 include_distance=True):
        self.station_locations = station_locations
        self.attributes_to_add = attributes_to_add
        self.include_distance = include_distance

    def fit(self, X, y=None):
        return self

    def transform(self, X, y):
        X['day'] = pd.to_datetime(X['day'])
        self.full = X.copy()
        for i, j in tqdm(
                zip(self.attributes_to_add, range(len(self.attributes_to_add))), 
                total=len(self.attributes_to_add)):
            self.full = pd.merge(self.full, y[j][i], how='inner',
                                 left_on='day', right_on='DATA',
                                 suffixes=("_" + str(j-1), "_" + str(j)))

        if self.include_distance:
            for j in tqdm(range(len(self.attributes_to_add))):
                self.add_distance(j)

        return self.full

    def add_distance(self, index):
        lat = self.station_locations[
                self.station_locations['Station'] == np.unique(
                        self.full['ESTACIO_' + str(index)])[0]][
                        'Latitude'].values
        long = self.station_locations[
                self.station_locations['Station'] == np.unique(
                        self.full['ESTACIO_' + str(index)])[0]][
                        'Longitude'].values

        self.full['DIST_' + str(index)] = self.full.apply(
                lambda x: distance.geodesic(
                        (x['LAT'], x['LON']), (lat, long)).km, axis=1)


def read_real_files(direc="./climateChallengeData/real"):
    """
    Takes the files in the folder real and groups them in a sole pandas
    THis dataframe will be the base for adding more features and make the
    training val and test division
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
    gridPoints.to_csv(file_name)


def create_idx(df):
    df['nxny'] = df.apply(lambda x: str(int(x['nx'])).strip() + str(int(x['ny'])).strip(), axis=1)


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parsed = parse_arguments(parser)

    download_files()
    full_real = read_real_files()
    official_stations_daily = read_official_stations()
    official_stations_latlon = pd.read_csv("./climateChallengeData/officialStations.csv")
    grid_points = pd.read_csv("./climateChallengeData/grid2latlon.csv")

    y_columns = ['T_MEAN']
    x_columns = ['day', 'ny', 'nx']

    X = DataFrameSelector(x_columns).transform(full_real)
    y = DataFrameSelector(y_columns).transform(full_real)

    create_idx(X)
    create_idx(grid_points)
    create_idx(official_stations_latlon)

    if parsed.comp_dist:
        compute_distances(official_stations_latlon, grid_points)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        shuffle=True)

    official_attr = [['DATA', 'Tm', 'Tx', 'Tn', 'ESTACIO'],
                     ['DATA', 'Tm', 'Tx', 'Tn', 'ESTACIO'],
                     ['DATA', 'Tm', 'Tx', 'Tn', 'ESTACIO'],
                     ['DATA', 'Tm', 'Tx', 'Tn', 'ESTACIO']]

    X_train_complete = official_station_daily_adder(
            official_attr,
            official_stations_latlon,
            include_distance=False).transform(
            X_train, official_stations_daily)

    print(X_train_complete.head(20))
