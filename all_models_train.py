#!/usr/bin/env python3

"""
Offline Training and Prediction with Online Models on Multiple Datasets

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : March 2023
Description : This script:
              - Combine multiple datset together in a unique one to train the
                models on multiple setup runs as if it has been just one run.
              - Processes power and performance traces read from local files.
              - Trains power (pl and ps) and performance online models.
              - Predicts with the trained models.
              All is done at offline with the idea of simplifying and speeding
              up the debugging and fine-tunning of the online models.
              The traces could be obtained in the real environment and the
              models could be trained and tested to fine-tune them easily.
"""


import time
import sys
import argparse
import threading
import pickle
import pandas as pd

from remote_execution.paper_results.always_training.incremental_learning import online_models as om_all
from remote_execution.paper_results.just_one_train.incremental_learning import online_models as om_one
from remote_execution.paper_results.adaptative_train.incremental_learning import online_models as om_adapt


# DEBUG: no dataframe printing column limit
pd.set_option('display.max_columns', None)

# Parse arguments
parser = argparse.ArgumentParser()

# Indicate which board has been used
parser.add_argument("board",
                    choices=["ZCU", "PYNQ"],
                    help="Type of board used")

# This is the correct way to handle accepting multiple arguments.
# '+' == 1 or more.
# '*' == 0 or more.
# '?' == 0 or 1.
# An int is an explicit number of arguments to accept.
# Indicate the path of the file containing the datasets.
parser.add_argument(
    '-d',
    dest="datasets_path",
    nargs='+',
    help='<Required> Paths to the input datasets',
    required=True
    )

# Indicate the name of the file containing the dataframe used to tsst
parser.add_argument(
    "-t",
    dest="test_dataset_path",
    default=None,
    help="test observations file path"
    )

args = parser.parse_args(sys.argv[1:])

# Create thread safe lock
lock_all = threading.Lock()
lock_one = threading.Lock()
lock_adapt = threading.Lock()

# Initialize the online models
online_models_all = om_all.OnlineModels(board=args.board, lock=lock_all)
online_models_one = om_one.OnlineModels(board=args.board, lock=lock_one)
online_models_adapt = om_adapt.OnlineModels(board=args.board, lock=lock_adapt)
print("Online Models have been successfully initialized.")

# Time measurement logic
t_start = time.time()

# TODO: TEST - Get first dataset

print(args.datasets_path)
input_path = args.datasets_path[0]
print(f"Input dataset path -> '{input_path}'")

# Read and concatenate dataframes
multiple_df = []

dataset_length_sum = 0
for dir_count, tmp_path in enumerate(args.datasets_path):
    multiple_df.append(pd.read_pickle(tmp_path))
    dataset_length_sum += len(multiple_df[dir_count])
    print(f"Dataset #{dir_count} - len: {len(multiple_df[dir_count])} - total: {dataset_length_sum}")

#exit(1)

# test flip
#multiple_df = multiple_df[::-1]

print(f"len: {len(multiple_df[0])}")

print("\n\nMultiple:\n")
print(multiple_df)

combined_df = pd.concat(multiple_df, axis=0, ignore_index=True)

print("\n\nCombined:\n")
print(combined_df)

train_df = combined_df

# Read train dataframe
# train_df = pd.read_pickle(input_path)

train_number_raw_observations = len(train_df)
# remove nan (i this it happends due to not using mutex for introducing cpu_usage)
train_df = train_df.dropna()

train_number_observations = len(train_df)

print(f"Train NaN rows: {train_number_raw_observations - train_number_observations}")

print(f"test: {args.test_dataset_path}")

if args.test_dataset_path is not None:
    # Read train dataframe
    test_df = pd.read_pickle(args.test_dataset_path)

    test_number_raw_observations = len(test_df)
    # remove nan (i this it happends due to not using mutex for introducing cpu_usage)
    test_df = test_df.dropna()

    test_number_observations = len(test_df)

    print(f"Test NaN rows: {test_number_raw_observations - test_number_observations}")
else:
    test_df = train_df

print(f"train_df: {train_df} | test_df: {test_df}")

def pruebas(train_df, online_models, data_save_file_name):
    ################
    ## the datasets to mimic how the setup would send traces in batches
    ################

    # Define the size of each sub-DataFrame
    SIZE = 200  # Change m to the desired size

    # Por que falla la training region cuando m no multiplo de 200

    # Calculate the number of chunks
    num_chunks = (len(train_df) + SIZE - 1) // SIZE

    # Local variables
    iteration = 0
    next_operation_mode = "train"
    wait_obs = 0
    # Loop over the DataFrame in chunks
    for i in range(num_chunks):

        # Calculate the start and end indices for each chunk
        start_index = i * SIZE
        # Take the minimum to avoid going beyond the length of the DataFrame
        end_index = min((i + 1) * SIZE, len(train_df))

        # Extract the sub-DataFrame
        sub_df = train_df.iloc[start_index:end_index]

        print(SIZE, num_chunks, i, start_index, end_index, len(sub_df))

        # Simulate what the setup would do if still in idle mode
        # Basically it runs x times till the wait obs elapse
        if next_operation_mode == "idle":
            # Simulation of the setup
            # If we have to wait any obs(what the setup would do)...
            # We decrement the wait a batch lenght an go to next one
            # It could happend that wait_obs is less than len(sub_df) so some obs of sub_df will be
            # wasted. But the setup would do something similar since it cannot perfectly cound obs...
            if wait_obs > 0:
                wait_obs -= len(sub_df)
                # Update the training metric, just so it can be properly compared with traditional models for the paper
                # Have in mind that in real scenario those traces are not generated
                online_models.idle_update_metric(sub_df)
                continue

        # Train/test from each sub_df
        iteration, next_operation_mode, wait_obs = online_models.update_models_zcu(sub_df, iteration)

        # Tell the setup to get measurements (when in train or test) or wait (when in idle phase)
        # This is simulated in previous part

    # TODO: Remove
    # When there are no more obs the system is either in train or test mode.
    # We need fill the last test/train_region list with the actual iteration
    if online_models._top_power_model._training_monitor.operation_mode == "train":
        online_models._top_power_model._training_monitor.train_train_regions[-1].append(iteration-1)
    elif online_models._top_power_model._training_monitor.operation_mode == "test":
        online_models._top_power_model._training_monitor.test_test_regions[-1].append(iteration-1)
    if online_models._bottom_power_model._training_monitor.operation_mode == "train":
        online_models._bottom_power_model._training_monitor.train_train_regions[-1].append(iteration-1)
    elif online_models._bottom_power_model._training_monitor.operation_mode == "test":
        online_models._bottom_power_model._training_monitor.test_test_regions[-1].append(iteration-1)
    if online_models._time_model._training_monitor.operation_mode == "train":
        online_models._time_model._training_monitor.train_train_regions[-1].append(iteration-1)
    elif online_models._time_model._training_monitor.operation_mode == "test":
        online_models._time_model._training_monitor.test_test_regions[-1].append(iteration-1)
    ###############

    # Save the models training monitor
    with open(f"./model_error_figures/{data_save_file_name}_training_monitors.pkl", 'wb') as file:
        tmp_var = [
            online_models._top_power_model._training_monitor,
            online_models._bottom_power_model._training_monitor,
            online_models._time_model._training_monitor
            ]
        pickle.dump(tmp_var, file)

    # Save the actual models
    with open(f"./model_error_figures/{data_save_file_name}_models.pkl", 'wb') as file:
        tmp_var = [
            online_models._top_power_model,
            online_models._bottom_power_model,
            online_models._time_model
            ]
        pickle.dump(tmp_var, file)

    return online_models


print("Pruebas_all")
online_models_all = pruebas(train_df, online_models_all, "all")
print("Pruebas_one")
online_models_one = pruebas(train_df, online_models_one, "one")
print("Pruebas_adapt")
online_models_adapt = pruebas(train_df, online_models_adapt, "adapt")


# Print training stages
print("Resultados all:\n")
online_models_all.print_training_monitor_info()
print("Resultados one:\n")
online_models_one.print_training_monitor_info()
print("Resultados adapt:\n")
online_models_adapt.print_training_monitor_info()

exit()

tmp_top_metric, tmp_bottom_metric, tmp_time_metric = online_models.get_metrics()
print(
    f"Training Metrics: {tmp_top_metric} (top) | {tmp_bottom_metric} (bottom)"
    f" | {tmp_time_metric} (time)"
)

# Seguir comentando

online_models.test(train_df)
if args.test_dataset_path is not None:
    online_models.test(test_df)

# Time measurement logic
t_end = time.time()

# Print useful information
print("Total Elapsed Time (s):", t_end-t_start, (t_end-t_start)/train_number_observations)
print("Number of trainings:", len(train_df))

if args.board == "ZCU":
    tmp_top_metric, tmp_bottom_metric, tmp_time_metric = \
        online_models.get_metrics()
    print(
        f"Training Metrics: {tmp_top_metric} (top) | {tmp_bottom_metric} (bottom) "
        f"| {tmp_time_metric} (time)"
    )
elif args.board == "PYNQ":
    tmp_power_metric, tmp_time_metric = online_models.get_metrics()
    print(f"Training Metrics: {tmp_power_metric} (power) | {tmp_time_metric} (time)")

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

    print(f"top: {top_power_prediction} | bot: {bottom_power_prediction} | time: {time_prediction}")

elif args.board == "PYNQ":
    power_prediction, time_prediction = online_models.predict_one(features)

    print("features:", features)

    print(f"power: {power_prediction} | time: {time_prediction}")
