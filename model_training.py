#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 20:46:43 2019

@author: juan

File to train the models.
"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_complete, y,
                                                    test_size=test_size,
                                                    shuffle=True)


if __name__ == "__main__":
    pass