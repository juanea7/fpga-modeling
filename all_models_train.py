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
import os
import argparse
import threading
import pickle
import pandas as pd

from incremental_learning import online_models as om


# DEBUG: no dataframe printing column limit
pd.set_option('display.max_columns', None)

# Parse arguments
parser = argparse.ArgumentParser()

# Indicate which board has been used
parser.add_argument("board",
                    choices=["ZCU", "PYNQ", "AU250"],
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
# Indicate the path to store the outputs of the script.
parser.add_argument(
    '-o',
    dest="outputs_path",
    help='<Required> Paths to store the outputs',
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

# Set the board
if args.board == "ZCU":
    board = {"power": {"rails": "dual",
                    "process": "zcu"},
            "traces": {"num_signals": 16,
                        "freq_MHz": 100},
            "arch": "64bit"}
elif args.board == "PYNQ":
    board = {"power": {"rails": "mono",
                    "process": "pynq"},
            "traces": {"num_signals": 8,
                        "freq_MHz": 100},
            "arch": "32bit"}
elif args.board == "AU250":
    board = {"power": {"rails": "mono",
                    "process": "au250"},
            "traces": {"num_signals": 32,
                        "freq_MHz": 100},
            "arch": "64bit"}
else:
    raise ValueError(F"Board not supported: {args.board}")

# Create thread safe lock
lock_all = threading.Lock()
lock_one = threading.Lock()
lock_adapt = threading.Lock()

# Initialize the online models
online_models_all = om.OnlineModels(power_rails=board["power"]["rails"], lock=lock_all, train_mode="always_train", capture_all_traces=True)
online_models_one = om.OnlineModels(power_rails=board["power"]["rails"], lock=lock_one, train_mode="one_train", capture_all_traces=True)
online_models_adapt = om.OnlineModels(power_rails=board["power"]["rails"], lock=lock_adapt, train_mode="adaptive", capture_all_traces=True)
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
                online_models.idle_update_metrics(sub_df)
                continue

        # Train/test from each sub_df
        iteration, next_operation_mode, wait_obs = online_models.update_models(sub_df, iteration)

        # Tell the setup to get measurements (when in train or test) or wait (when in idle phase)
        # This is simulated in previous part

    # TODO: Remove
    # When there are no more obs the system is either in train or test mode.
    # We need fill the last test/train_region list with the actual iteration
    for model in online_models._models:
        if model._training_monitor.operation_mode == "train":
            model._training_monitor.train_train_regions[-1].append(iteration-1)
        elif model._training_monitor.operation_mode == "test":
            model._training_monitor.test_test_regions[-1].append(iteration-1)
    ###############


    # Save the models training monitor
    if not os.path.exists(args.outputs_path):
        os.makedirs(args.outputs_path)
    with open(f"{args.outputs_path}/{data_save_file_name}_training_monitors.pkl", 'wb') as file:
        tmp_var = [model._training_monitor for model in online_models._models]
        pickle.dump(tmp_var, file)

    # Save the actual models
    with open(f"{(args.outputs_path)}/{data_save_file_name}_models.pkl", 'wb') as file:
        tmp_var = [model for model in online_models._models]
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
online_models_all.print_training_monitor_info(args.outputs_path)
print("Resultados one:\n")
online_models_one.print_training_monitor_info(args.outputs_path)
print("Resultados adapt:\n")
online_models_adapt.print_training_monitor_info(args.outputs_path)
