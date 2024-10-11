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
import os
import sys

from incremental_learning_just_dual import online_data_processing as odp
from incremental_learning_just_dual import online_models as om

from pprint import pprint


def loop_progress_bar(actual_iteration,
                      total_iterations,
                      indentation=0,
                      show_iteration=False):
    """Generates a progress bar to indicate the percentage of the loop already
    executed
    """

    # Calculate completed ratio (0-1)
    ratio = (actual_iteration + 1) / total_iterations
    # Clear the output
    sys.stdout.write('\r')
    # Write progress
    if show_iteration:
        sys.stdout.write("{0}[{1:<20}] {2:d}% ({3}/{4})".format(
            '\t' * indentation,
            '=' * int(20 * ratio),
            int(100 * ratio),
            actual_iteration,
            total_iterations)
        )
    else:
        sys.stdout.write("{0}[{1:<20}] {2:d}%".format(
            '\t' * indentation,
            '=' * int(20 * ratio),
            int(100 * ratio))
        )
    # Force writting to stdout
    sys.stdout.flush()


# DEBUG: no dataframe printing column limit
pd.set_option('display.max_columns', None)

# Initialize the online models
online_models = om.OnlineModels()
print("Online Models have been successfully initialized.")

# Set the path to the online.bin and traces files
output_data_path = "outputs"
traces_data_path = "traces"

# Useful local variables
t_interv = 0
i = 0

# Total amount of iterations (for the progress bar)
num_iters = len(os.listdir(output_data_path))
print("Progress:")

# Counter of non-meaningful observations
num_bad_obs = 0

# Keep observations
all_obs_df = None

# debug
wrong_traces = []

# Keep processsing files undefinatelly
# (test with finite number of iterations)
for filename in os.listdir(output_data_path):

    if "online" not in filename:
        print("no online:", filename)
        continue
    # Show progress bar
    loop_progress_bar(i, num_iters, 0, True)
    if i == 0:
        # Time measurement logic
        t_start = time.time()

    t_inter0 = time.time()

    # Generate the next online_info files path
    curr_output_path = os.path.join(output_data_path,
                                    "online_info_{}.bin".format(i))
    curr_power_path = os.path.join(traces_data_path,
                                   "CON_{}.BIN".format(i))
    curr_traces_path = os.path.join(traces_data_path,
                                    "SIG_{}.BIN".format(i))

    i += 1

    # Generate observations for this particular online_info.bin file
    generated_obs = \
        odp.generate_observations_from_online_data(
            curr_output_path,
            curr_power_path,
            curr_traces_path
        )

    # Check if this monitoring windows contains observations
    # In case there are no observations just move to next window
    if len(generated_obs) < 1:
        # print("No meaningful observations")
        t_inter1 = time.time()
        t_interv += t_inter1-t_inter0
        num_bad_obs += 1
        continue
    else:
        for obs in generated_obs:
            if float(obs[2]) < 1.0:
                wrong_traces.append((i - 1, obs))

    # Create dataframe just for this online_data file
    df = odp.generate_dataframe_from_observations(generated_obs)

    if all_obs_df is None:
        all_obs_df = df
    else:
        all_obs_df = pd.concat([df, all_obs_df], ignore_index=True)

    # DEBUG: Checking with observations are generated
    # print(curr_output_path)
    # print(traces_path)
    # print(curr_traces_file)
    # print("Observation:\n", df)

    # Learn batch with the online models
    online_models.train(df)

    # tmp_top_metric, tmp_bottom_metric, tmp_time_metric = \
    #     online_models.get_metrics()
    # print(
    #     "Training Metrics: {} (top) | {} (bottom)"
    #     " | {} (time)".format(tmp_top_metric,
    #                           tmp_bottom_metric,
    #                           tmp_time_metric
    #                           )
    # )

    t_inter1 = time.time()
    t_interv += t_inter1-t_inter0

# Time measurement logic
t_end = time.time()

# Print useful information
if i > 0:  # Take care of division by zero
    print("Interval Elapsed Time (s):", t_interv, (t_interv)/i)
    print("Total Elapsed Time (s):", t_end-t_start, (t_end-t_start)/i)
    print("Number of trainings:", i)

else:
    print("No processing was made")

print("Total of observations: {} | Non-meaningful: {}".format(i, num_bad_obs))

tmp_top_metric, tmp_bottom_metric, tmp_time_metric = \
    online_models.get_metrics()
print(
    "Training Metrics: {} (top) | {} (bottom) | {} (time)".format(
        tmp_top_metric,
        tmp_bottom_metric,
        tmp_time_metric
    )
)

# Set observations file path
# (there because is executed from ssh without permission in folder outputs)
# Solvable by changing the ownership of that folder
observations_path = "runtime_observations.pkl"

# Save the dataframe in a file
all_obs_df.to_pickle(observations_path)

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

print("All observations:\n{}".format(all_obs_df))

print("wrong traces: (len: {})".format(len(wrong_traces)))
pprint(wrong_traces)
