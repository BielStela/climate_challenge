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
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np

np.random.seed(42)
test_size = 0.2

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
    
#
#class official_station_daily_adder(BaseEstimator, TransformerMixin):
#    
#    """
#    Class for adding attributes from official stations to the values predicted 
#    from the model (real values). The data is the joint for the moment
#    """
#    
#    def __init__(self, attributes_to_add):
#        self.attributes_to_add = attributes_to_add
#
#    def fit(self, X, y=None):
#        return self
#    
#    def transform(self,X, y):
#        for index, row in X.iterrows():
#            date = pd.to_datetime(row['day'], format='%Y-%m-%d')
#            y[y['DATA'] == date]

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

def read_official_stations(name = "./climateChallengeData/data_S2_S3_S4.xlsx"):
    return pd.read_excel(name, delimeter=";", sheet_name=[0,1,2,3])
    
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
        with open(path.join(direc, folder_name, name), 'wb')  as f:
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
    
    download_files()
    full_real = read_real_files()
    official_stations_daily = read_official_stations()
    
    y_columns = ['T_MEAN']
    x_columns = ['day','LAT','LON']
    
    X = DataFrameSelector(x_columns).transform(full_real)
    y = DataFrameSelector(y_columns).transform(full_real)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)

    """
    
    """
    print(official_station_daily[1])
    
    #TODO:
    # make class or fucntion to add official stations values to real values
    #joining them with date values. FOr example, if I have one row with date x 
    #I will take the values of day x in official stations and put them as features
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    