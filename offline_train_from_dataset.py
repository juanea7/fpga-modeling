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
from urllib.parse import non_hierarchical
import pandas as pd
#import os
import sys
import argparse
import numpy as np
from scipy.stats import skew, boxcox

from incremental_learning import online_models as om

import river

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

# Indicate the name of the file containing the dataframe used to tsst
parser.add_argument("-t", dest="test_dataset_path", default=None,
                    help="test observations file path")

args = parser.parse_args(sys.argv[1:])

# Initialize the online models
online_models = om.OnlineModels(board=args.board)
print("Online Models have been successfully initialized.")

# Time measurement logic
t_start = time.time()

# Read train dataframe
train_df = pd.read_pickle(args.input_path)

train_number_raw_observations = len(train_df)
# remove nan (i this it happends due to not using mutex for introducing cpu_usage)
train_df = train_df.dropna()

train_number_observations = len(train_df)

print("Train NaN rows: {}".format(train_number_raw_observations - train_number_observations))

print("test: {}".format(args.test_dataset_path))

if args.test_dataset_path is not None:
    # Read train dataframe
    test_df = pd.read_pickle(args.test_dataset_path)

    test_number_raw_observations = len(test_df)
    # remove nan (i this it happends due to not using mutex for introducing cpu_usage)
    test_df = test_df.dropna()

    test_number_observations = len(test_df)

    print("Test NaN rows: {}".format(test_number_raw_observations - test_number_observations))
else:
    test_df = train_df

print("train_df: {} | test_df: {}".format(train_df, test_df))

# Test
#df['Top power'] = df['Top power'].apply(lambda x: x*10)
#df['Bottom power'] = df['Bottom power'].apply(lambda x: x*10)
#df['Time'] = df['Time'].apply(lambda x: x*10)

# DEBUG: Checking with observations are generated
# print(curr_output_path)
# print(traces_path)
# print(curr_traces_file)

# Test less features
#print(df)

#df = df.apply(lambda x: x if x.name == 'Main' or x.name == 'Top power' or x.name == 'Bottom power' or x.name == 'Time' else x.apply(lambda y: 1 if y > 0 else y))

#print(df)



#print("len observations: {}\n".format(number_observations))
#print("Observations:\n", df)

#test_skew = df["Time"].to_numpy()
#print(test_skew)
#print("skew:", skew(test_skew))
##test_log = np.log(test_skew)
##test_log = np.sqrt(test_skew)
#test_log = boxcox(test_skew)[0]
#print("log skew:", skew(test_log))
#
##df["Time"] = np.log(df["Time"].to_numpy())
##df["Time"] = np.sqrt(df["Time"].to_numpy())
#df["Time"] = boxcox(df["Time"])[0]
#df["Time"] = df["Time"].apply(lambda x: x*10)

# Learn batch with the online models
online_models.train(train_df)

tmp_top_metric, tmp_bottom_metric, tmp_time_metric = \
    online_models.get_metrics()
print(
    "Training Metrics: {} (top) | {} (bottom)"
    " | {} (time)".format(tmp_top_metric,
                          tmp_bottom_metric,
                          tmp_time_metric
                          )
)

online_models.test(train_df)
if args.test_dataset_path is not None:
    online_models.test(test_df)

# Time measurement logic
t_end = time.time()

# Print useful information
print("Total Elapsed Time (s):",
      t_end-t_start, (t_end-t_start)/train_number_observations)
print("Number of trainings:", len(train_df))

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
    "user": 58.08, 
    "kernel": 33.33,  
    "idle": 8.59,
    "Main": 2,
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
