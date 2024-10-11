#!/usr/bin/env python3

"""
Grid Search of Adaptative Parameters for Models Retrain and Stop Train Strats

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : October 2023
Description : This scripts perform a grid search varying the parameters that
              control the re-train and stop training strategies of all the
              adaptative models.
"""


import time
from urllib.parse import non_hierarchical
import pandas as pd
import sys
import argparse

from incremental_learning import online_models as om


# DEBUG: no dataframe printing column limit
pd.set_option('display.max_columns', None)

# Parse arguments
parser = argparse.ArgumentParser()

# Indicate which board has been used
parser.add_argument("board",
                    choices=["ZCU", "PYNQ"],
                    help="Type of board used")
                    
# Indicate the path of the file containing the datasets.
parser.add_argument('-d', dest="datasets_path", nargs='+', help='<Required> Paths to the input datasets', required=True)

# Indicate the name of the file containing the dataframe used to tsst
parser.add_argument("-t", dest="test_dataset_path", nargs='+', help="test observations file path", default=None)
                    
args = parser.parse_args(sys.argv[1:])

# Initialize the online models
online_models = om.OnlineModels(board=args.board)
print("Online Models have been successfully initialized.")

# Time measurement logic
t_start = time.time()

# Read and concatenate dataframes
multiple_df = []
dataset_length_sum = 0
for dir_count, tmp_path in enumerate(args.datasets_path):
    multiple_df.append(pd.read_pickle(tmp_path))
    dataset_length_sum += len(multiple_df[dir_count])
    print("Dataset #{} - len: {} - total: {}".format(dir_count, len(multiple_df[dir_count]), dataset_length_sum))

print("len: {}".format(len(multiple_df[0])))

# Combine dataset
combined_df = pd.concat(multiple_df, axis=0, ignore_index=True)
train_df = combined_df

train_number_raw_observations = len(train_df)
# remove nan (i this it happends due to not using mutex for introducing cpu_usage)
train_df = train_df.dropna()

train_number_observations = len(train_df)

print("Train NaN rows: {}".format(train_number_raw_observations - train_number_observations))

print("test: {}".format(args.test_dataset_path))

# Get test dataset if testing is enabled
if args.test_dataset_path is not None:
    # Read train dataframe
    test_df = pd.read_pickle(args.test_dataset_path)

    test_number_raw_observations = len(test_df)
    # remove nan (i this it happends due to not using mutex for introducing cpu_usage)
    test_df = test_df.dropna()

    test_number_observations = len(test_df)

    print("Test NaN rows: {}".format(test_number_raw_observations - test_number_observations))

#print("train_df: {} | test_df: {}".format(train_df, test_df))

# Do a grid search for testing different parameters for customizing the adaptative models
grid_search_params = {
    "train": {
        "nominal_obs_btw_validation": [1000],
        "obs_btw_validation_reduction_factor": [0, 0.2],
        "validation_threshold": [3],
        "stable_training_error_threshold": [3]
    },
    "test": {
        "obs_btw_test": [1000],
        "nominal_obs_to_test": [200],
        "test_threshold": [6],
        "obs_to_test_reduction_factor": [0, 0.2],
        "significant_error_variation_threshold": [2]
    }
}

# Grid search
# online_models.grid_search_train(train_df, grid_search_params)
online_models.grid_search_train_multiprocessing(train_df, grid_search_params)

tmp_top_metric, tmp_bottom_metric, tmp_time_metric = \
    online_models.get_metrics()
print(
    "Training Metrics: {} (top) | {} (bottom)"
    " | {} (time)".format(tmp_top_metric,
                          tmp_bottom_metric,
                          tmp_time_metric
                          )
)

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
