#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 18:36:01 2019

@author: juan
"""

from utils_full import add_df, drop_function
from ax.core import SearchSpace, ChoiceParameter, ParameterType, Experiment
from utils_reader import read_real_files, create_idx
from ax import Models, Metric, OptimizationConfig, Objective, Runner
from ax.core.data import Data
import numpy as np
import tqdm as tqdm
import pandas as pdq
from utils_train import classify_one_idx
from sklearn.metrics import mean_squared_error
from sklearn.linear import LinearRegression
from sklearn.model_selection import KFold

OFFICIAL_ATTR_2 = [['DATA', 'Tm'],
                   ['DATA', 'Tm'],
                   ['DATA', 'Tm'],
                   ['DATA', 'Tm']]

def main():

    df = load_basic()
    print(df.columns)


def load_basic(official_attr=OFFICIAL_ATTR_2):
    real = read_real_files()
    create_idx(real)
    df = add_df(real, OFFICIAL_ATTR_2)
    drop_function(df, "ESTACIO_")
    drop_function(df, "DATA")
    df.drop(columns=['ndays', 'nx', 'ny', 'LAT', 'LON', 'T_MIN', 'T_MAX'],
            inplace=True)
    return df


def performance(params, df):
    df = add_point(params, df)
    return return_performance(df)


def add_point(params, df):
    idx = params['x'] + params['y']
    df_selected = df[df['idx'] == idx]
    df_full = df.merge(df_selected, how='inner',
                       left_on='day', right_on='day',
                       suffixes=("", "_1"))
    return df_full


def return_performance(X):

    X_t = X[X.columns[(X.columns != 'T_MEAN') &
                      (X.columns != 'idx') & (X.columns != 'day')]].values
    y = X['T_MEAN'].values
    results = []
    fold = KFold(n_splits=5, shuffle=True)

    for train_idx, test_idx in fold.split(X_t):
        X_train, X_test = X_t[train_idx], X_t[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        linear = LinearRegression(n_jobs=-1)
        linear.fit(X_train, y_train)
        y_pred = linear.predict(X_test)

        results.append(mean_squared_error(y_test, y_pred))

    return np.mean(results)


class LRMetric(Metric):
    def fetch_trial_data(self, trial):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "mean": performance(params),
                "sem": 0.0,
                "trial_index": trial.index,
            })
        return Data(df=pd.DataFrame.from_records(records))

class MyRunner(Runner):
    def run(self, trial):
        return {"name": str(trial.index)}

def bo_loop(df, nx, ny):
    range_x = ChoiceParameter(name='x', values=set(map(str, nx)),
                              parameter_type=ParameterType.STRING)
    range_y = ChoiceParameter(name="y", values=set(map(str, ny)),
                              parameter_type=ParameterType.STRING)
    space = SearchSpace(parameters=[range_x, range_y])

    experiment = Experiment(name="experiment_one_cell",
                            search_space=space)

    sobol = Models.SOBOL(search_space=experiment.search_space)
    generator_run = sobol.gen(100)

    optimization_config = OptimizationConfig(objective=Objective(
        metric=LRMetric(name="lr"),
        minimize=True,
        ),
    )

    experiment.optimization_config = optimization_config
    experiment.runner = MyRunner()
    experiment.new_batch_trial(generator_run=generator_run)
    experiment.trials[0].run()
    data = experiment.fetch_data()

    gpei = Models.BOTORCH(experiment=experiment, data=data)
    generator_run = gpei.gen(1000)
    experiment.new_batch_trial(generator_run=generator_run)

    experiment.trials[-1].run()
    data = experiment.fetch_data()
    
if __name__ == "__main__":
    main()