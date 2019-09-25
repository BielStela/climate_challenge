#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 18:36:01 2019

@author: juan
"""

from utils_full import add_df, drop_function
from ax.core import SearchSpace, RangeParameter, ParameterType, Experiment
from utils_reader import read_real_files, create_idx
from ax import Models, Metric, OptimizationConfig, Objective, Runner, save
from ax.core.data import Data
from ax.storage.metric_registry import register_metric
from ax.storage.runner_registry import register_runner
import numpy as np
from tqdm import tqdm
import pandas as pd
from utils_train import classify_one_idx
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from functools import partial


OFFICIAL_ATTR_2 = [['DATA', 'Tm'],
                   ['DATA', 'Tm'],
                   ['DATA', 'Tm'],
                   ['DATA', 'Tm']]


def main():

    df, index, grid = load_basic()

    for i in range(10):
        bo_loop(df, index, grid, "experiment_" + str(i) + ".json")


def load_basic(official_attr=OFFICIAL_ATTR_2):
    real = read_real_files()
    grid_points = pd.read_csv("./climateChallengeData/grid2latlon.csv")
    create_idx(grid_points)
    i = len(grid_points)
    create_idx(real)
    df = add_df(real, OFFICIAL_ATTR_2)
    drop_function(df, "ESTACIO_")
    drop_function(df, "DATA")

    df.drop(columns=['ndays', 'nx', 'ny', 'LAT', 'LON', 'T_MIN', 'T_MAX'],
            inplace=True)
    return df, i, grid_points


def performance(df, grid, params):
    df_opt = add_point(params, df, grid)
    return return_performance(df_opt)


def add_point(params, df, grid):
    index = params['index']
    idx = grid.loc[index, 'idx']
    df_selected = df[df['idx'] == str(idx)]
    df_selected = df_selected.loc[:, ['day', 'T_MEAN']]
    jj = len(df.columns)
    df_full = df.merge(df_selected, how='inner',
                       left_on='day', right_on='day',
                       suffixes=("", "_" + str(jj + 1)))
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

    def __init__(self, partial_performance, name):
        self.partial_performance = partial_performance
        self._name = name
        self.lower_is_better = True

    def fetch_trial_data(self, trial):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "mean": self.partial_performance(params),
                "sem": 0.0,
                "trial_index": trial.index,
            })
        return Data(df=pd.DataFrame.from_records(records))


class MyRunner(Runner):
    def run(self, trial):
        return {"name": str(trial.index)}


def bo_loop(df, index, grid, name):
    range_x = RangeParameter(name='index', lower=0, upper=index-1,
                             parameter_type=ParameterType.INT)
    space = SearchSpace(parameters=[range_x])

    experiment = Experiment(name="experiment_one_cell",
                            search_space=space)

    optimization_config = OptimizationConfig(objective=Objective(
        metric=LRMetric(partial(performance, df, grid), name="lr"),
        minimize=True,
        ),
    )

    experiment.optimization_config = optimization_config
    experiment.runner = MyRunner()

    sobol = Models.SOBOL(search_space=experiment.search_space)
    for i in tqdm(range(1)):
        generator_run = sobol.gen(1)
        experiment.new_trial(generator_run=generator_run)
        experiment.trials[i].run()
        data = experiment.fetch_data()

    for i in tqdm(range(1, 1 + 5)):
        gpei = Models.BOTORCH(experiment=experiment, data=data)
        generator_run = gpei.gen(1)
        experiment.new_trial(generator_run=generator_run)
        experiment.trials[i].run()
        data = experiment.fetch_data()


    print(data.df)




if __name__ == "__main__":
    main()