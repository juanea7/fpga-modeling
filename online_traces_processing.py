#!/usr/bin/env python3

"""
Processing of online traces into a dataframe as a pickle file

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : May 2023
Description : This script:
              - Processes power and performance traces read from local files.
              - Generates a pickle file containing a dataframe with all
                processed observations.

"""


import time
import pandas as pd
import os
import sys
import argparse

from incremental_learning import online_data_processing as odp

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

# Parse arguments
parser = argparse.ArgumentParser()

# Indicate which board has been used
parser.add_argument("board",
                    choices=["ZCU", "PYNQ", "AU250"],
                    help="Type of board used")

# Indicate the path to read traces from
parser.add_argument("input_dir",
                    help="Directory to read traces from")

# Indicate the name of the file containing the dataframe processed
parser.add_argument("-o", dest="output_path", default="output_obs",
                    help="Processed observations file path")

# Indicate if cpu usage or not (default is true)
parser.add_argument('--no-cpu-usage', action='store_true')

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

# Variable indicating cpu usage
cpu_usage_flag = not args.no_cpu_usage

print("CPU usage: {}".format(cpu_usage_flag))

# Set the path to the online.bin and traces files
output_data_path = args.input_dir + "/outputs"
traces_data_path = args.input_dir + "/traces"

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
                                    "online_{}.bin".format(i))
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
            curr_traces_path,
            board,
            cpu_usage=cpu_usage_flag
        )

    # Check if this monitoring windows contains observations
    # In case there are no observations just move to next window
    if len(generated_obs) < 1:
        # print("No meaningful observations")
        t_inter1 = time.time()
        t_interv += t_inter1-t_inter0
        num_bad_obs += 1
        continue
    # else:
    #     for obs in generated_obs:
    #         if float(obs[-3]) < 1.0:
    #             wrong_traces.append((i - 1, obs))

    # Create dataframe just for this online_data file
    df = odp.generate_dataframe_from_observations(generated_obs,
                                                  board,
                                                  cpu_usage=cpu_usage_flag)
    # print(df)

    if all_obs_df is None:
        all_obs_df = df
    else:
        all_obs_df = pd.concat([all_obs_df, df], ignore_index=True)

    # DEBUG: Checking with observations are generated
    # print(curr_output_path)
    # print(traces_path)
    # print(curr_traces_file)
    # print("Observation:\n", df)

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

# Set observations file path
# (there because is executed from ssh without permission in folder outputs)
# Solvable by changing the ownership of that folder
observations_path = args.output_path + ".pkl"

# Save the dataframe in a file
all_obs_df.to_pickle(observations_path)

print("All observations:\n{}".format(all_obs_df))

print("wrong traces: (len: {})".format(len(wrong_traces)))
pprint(wrong_traces)
