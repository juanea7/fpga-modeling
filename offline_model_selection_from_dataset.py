#!/usr/bin/env python3

"""
Offline Training and Prediction with Online Models

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : March 2023
Description : This script:
              - Processes power and performance traces read from local files.
              - Trains power (pl and ps) and performance online models.
              - Predicts with the trained models.
              All is done at offline with the idea of simplifying and speeding
              up the debugging and fine-tunning of the online models.
              The traces could be obtained in the real environment and the
              models could be trained and tested to fine-tune them easily.
"""


import time
import pandas as pd
#import os
import sys
import argparse

from incremental_learning import online_models as om

import river

from river import(
    ensemble
)

import copy

#from pprint import pprint


# DEBUG: no dataframe printing column limit
pd.set_option('display.max_columns', None)

# Parse arguments
parser = argparse.ArgumentParser()

# Indicate which board has been used
parser.add_argument("board",
                    choices=["ZCU", "PYNQ"],
                    help="Type of board used")

# Indicate the path of the file containing the dataset
parser.add_argument("input_path",
                    help="Path to the file containing the input dataset")

args = parser.parse_args(sys.argv[1:])

# Initialize the online models

# Hyper parameters exploration

test_top_models = [river.linear_model.LinearRegression(
                optimizer=river.optim.SGD(0.0001)
            )]

#test_top_models = river.utils.expand_param_grid(test_top_model, test_top_grid)
print("Number power models: {}".format(len(test_top_models)))


test_time_model = river.forest.ARFRegressor(seed=42, max_features=None)
test_time_grid = {
    'grace_period': [25],
    'n_models': [20],
    'aggregation_method': ['mean', 'median'],
    'disable_weighted_vote': [True, False],
    'merit_preprune': [True, False],
    'model_selector_decay': [0.95, 0.85, 0.75]
}
test_time_models = river.utils.expand_param_grid(test_time_model, test_time_grid)
print("Number time models: {}".format(len(test_time_models)))

for i in range(len(test_top_models)):

    print("\n\n\tIteration #{}\n\n".format(i))

    print("Power models:\n{}\n".format(test_top_models[i].__dict__))
    print("Time model:\n{}\n".format(test_time_models[i].__dict__))

    try_top_model = copy.deepcopy(test_top_models[i])
    try_bottom_model = copy.deepcopy(test_top_models[i])
    try_time_model = copy.deepcopy(test_time_models[i])

    online_models = om.OnlineModels(board=args.board, input_top_model=try_top_model, input_bottom_model=try_bottom_model, input_time_model=try_time_model)
    print("Online Models have been successfully initialized.")

    # Time measurement logic
    t_start = time.time()

    # Read dataframe
    df = pd.read_pickle(args.input_path)

    # DEBUG: Checking with observations are generated
    # print(curr_output_path)
    # print(traces_path)
    # print(curr_traces_file)
    number_observations = len(df)
    #print("len observations: {}\n".format(number_observations))
    #print("Observations:\n", df)

    # Learn batch with the online models
    online_models.train(df)

    tmp_top_metric, tmp_bottom_metric, tmp_time_metric = \
        online_models.get_metrics()
    print(
        "Training Metrics: {} (top) | {} (bottom)"
        " | {} (time)".format(tmp_top_metric,
                            tmp_bottom_metric,
                            tmp_time_metric
                            )
    )

    # Time measurement logic
    t_end = time.time()

    # Print useful information
    print("Total Elapsed Time (s):",
        t_end-t_start, (t_end-t_start)/number_observations)
    print("Number of trainings:", len(df))

    if args.board == "ZCU":
        tmp_top_metric, tmp_bottom_metric, tmp_time_metric = \
            online_models.get_metrics()
        print(
            "Training Metrics: {} (top) | {} (bottom) | {} (time)".format(
                tmp_top_metric,
                tmp_bottom_metric,
                tmp_time_metric
            )
        )
    elif args.board == "PYNQ":
        tmp_power_metric, tmp_time_metric = online_models.get_metrics()
        print(
            "Training Metrics: {} (power) | {} (time)".format(
                tmp_power_metric,
                tmp_time_metric
            )
        )

    # Random prediction
    features = {
        "aes": 0,
        "bulk": 0,
        "crs": 1,
        "kmp": 0,
        "knn": 1,
        "merge": 0,
        "nw": 1,
        "queue": 4,
        "stencil2d": 0,
        "stencil3d": 0,
        "strided": 0
    }

    if args.board == "ZCU":
        top_power_prediction, \
            bottom_power_prediction, \
            time_prediction = \
            online_models.predict_one(features)

        print("features:", features)

        print("top: {} | bot: {} | time: {}".format(
            top_power_prediction,
            bottom_power_prediction,
            time_prediction
            )
        )

    elif args.board == "PYNQ":
        power_prediction, time_prediction = online_models.predict_one(features)

        print("features:", features)

        print("pwer: {} | time: {}".format(power_prediction, time_prediction))
