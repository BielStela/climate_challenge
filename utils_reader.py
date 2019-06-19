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
import os
import pandas as pd
import urllib.request as req
import zipfile
import io



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
        return X[self.attribute_names].values


def read_real_files(direc="./climateChallengeData/real"):
    """
    Takes the files in the folder real and groups them in a sole pandas
    THis dataframe will be the base for adding more features and make the
    training val and test division
    """
    files = []
    for i in os.listdir(direc):
        name = os.path.join(direc, i)
        files.append(pd.read_csv(name))
    full_file = pd.concat(files)
    return full_file

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
        if not os.path.exists(os.path.join(direc, folder_name)):
            os.mkdir(os.path.join(direc, folder_name))
        with open(os.path.join(direc, folder_name, name), 'wb')  as f:
            f.write(content)
            
    if not os.path.exists(direc):
        url = "http://climate-challenge.herokuapp.com/climateApp/static/data/climateChallengeData.zip"
        url_sample1 = "http://climate-challenge.herokuapp.com/climateApp/static/sub_example/S1.csv"
        url_sample2 = "http://climate-challenge.herokuapp.com/climateApp/static/sub_example/S2.csv"
        
        print("Downloading \n", end="\r")

        with zipfile.ZipFile(io.BytesIO(give_request(url)), 'r') as z:

            print("Extracting data \n", end="\r")
            os.mkdir(direc)
            z.extractall(path=direc)
            
        save_sample(give_request(url_sample1), "S1.csv", direc)
        save_sample(give_request(url_sample2), "S2.csv", direc)
        
    else:
        print("Data already downloaded")
    
    
if __name__ == "__main__":
    
    download_files()
    read_real_files()
    