#!/usr/bin/env python3

"""
Run-Time Training and Prediction with Online Models

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : December 2023
Description : This script:
              - Processes power and performance traces received from an
                independent c-code process via disk-backed files or ram-backed
                memory-mapped regions (up to the user).
              - Trains power (pl and ps) and performance online models on
                demand from commands send via a socket from the independent
                c-code process.
              - Predicts with the trained models for the features receive from
                the independent c-code process via another socket.
              - Models are synchronized, training and testing simultaneously
              All is done at run-time with concurrent threads for training the
              models and predicting with them.

"""

import sys
import os
import argparse
import time
import struct
import threading
from datetime import datetime, timezone
# from multiprocessing import Process
import ctypes as ct
import pandas as pd
import river
import pickle
import itertools
from collections import Counter
import matplotlib as mpl
import matplotlib.pyplot as plt


from remote_execution.paper_results.adaptative_train.incremental_learning import online_models as om

from incremental_learning import online_data_processing as odp
from incremental_learning import inter_process_communication as ipc
#from incremental_learning import online_models as om
from incremental_learning import ping_pong_buffers as ppb
from incremental_learning import execution_modes_buffers as emb


class FeatureswCPUUsage(ct.Structure):
    """ Features with CPU Usage - This class defines a C-like struct """
    _fields_ = [
        ("user", ct.c_float),
        ("kernel", ct.c_float),
        ("idle", ct.c_float),
        ("aes", ct.c_uint8),
        ("bulk", ct.c_uint8),
        ("crs", ct.c_uint8),
        ("kmp", ct.c_uint8),
        ("knn", ct.c_uint8),
        ("merge", ct.c_uint8),
        ("nw", ct.c_uint8),
        ("queue", ct.c_uint8),
        ("stencil2d", ct.c_uint8),
        ("stencil3d", ct.c_uint8),
        ("strided", ct.c_uint8)
    ]

    def get_dict(self):
        """Convert to dictionary"""
        return dict((f, getattr(self, f)) for f, _ in self._fields_)


class FeatureswoCPUUsage(ct.Structure):
    """ Features without CPU Usage- This class defines a C-like struct """
    _fields_ = [
        ("aes", ct.c_uint8),
        ("bulk", ct.c_uint8),
        ("crs", ct.c_uint8),
        ("kmp", ct.c_uint8),
        ("knn", ct.c_uint8),
        ("merge", ct.c_uint8),
        ("nw", ct.c_uint8),
        ("queue", ct.c_uint8),
        ("stencil2d", ct.c_uint8),
        ("stencil3d", ct.c_uint8),
        ("strided", ct.c_uint8)
    ]

    def get_dict(self):
        """Convert to dictionary"""
        return dict((f, getattr(self, f)) for f, _ in self._fields_)


class PredictionZCU(ct.Structure):
    """ Prediction (ZCU) - This class defines a C-like struct """
    _fields_ = [
        ("top_power", ct.c_float),
        ("bottom_power", ct.c_float),
        ("time", ct.c_float)
    ]


class PredictionPYNQ(ct.Structure):
    """ Prediction (PYNQ) - This class defines a C-like struct """
    _fields_ = [
        ("power", ct.c_float),
        ("time", ct.c_float)
    ]


class MetricsZCU(ct.Structure):
    """ Errro Metrics (ZCU) - This class defines a C-like struct """
    _fields_ = [
        ("ps_power_error", ct.c_float),
        ("pl_power_error", ct.c_float),
        ("time_error", ct.c_float)
    ]


class MetricsPYNQ(ct.Structure):
    """ Errro Metrics (PYNQ) - This class defines a C-like struct """
    _fields_ = [
        ("power_error", ct.c_float),
        ("time_error", ct.c_float)
    ]


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


def show_model_data(online_models_data):
    """
    Process models output data and generate pertinent figures.

    Online Models Data: [
            online_models._top_power_model._training_monitor,
            online_models._bottom_power_model._training_monitor,
            online_models._time_model._training_monitor
            ]

    Temporal Data: [
        num_measurements,
        start_time_train/test,
        start_observation_index,
        end_time_train/test,
        end_observation_index
        ]

    """

    # Extract data from online models data
    top_training_monitor = online_models_data[0]
    bottom_training_monitor = online_models_data[1]
    time_training_monitor = online_models_data[2]

    # Matplotlib configuration
    mpl.rcParams['figure.figsize'] = (20, 12)
    # Remove top and right frame
    mpl.rcParams['axes.spines.left'] = True
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.bottom'] = True

    for model_index in range(3):
        # Create a 2x2 grid of subplots within the same figure
        fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True, constrained_layout=False)

        fig.supxlabel('Number of Observations')
        fig.suptitle('Error Metrics!!')

        if model_index == 0:
            # Add colored background spans to the plot (train)
            for xmin, xmax in top_training_monitor.train_train_regions:
                ax1.axvspan(
                    xmin, xmax, alpha=0.4,
                    color=top_training_monitor.train_train_regions_color,
                    zorder=0
                    )
            # Add colored background spans to the plot (test)
            for xmin, xmax in top_training_monitor.test_test_regions:
                ax1.axvspan(
                    xmin, xmax, alpha=0.4,
                    color=top_training_monitor.test_test_regions_color,
                    zorder=0
                    )
            # Plot model metrics
            ax1.plot(
                top_training_monitor.train_training_metric_history,
                label="adaptative_training_history",
                color='tab:green',
                zorder=2
                )
            # Set Y limit
            ax1.set_ylim([-0.5, 14.5])
        if model_index == 1:
            # Add colored background spans to the plot (train)
            for xmin, xmax in bottom_training_monitor.train_train_regions:
                ax1.axvspan(
                    xmin, xmax, alpha=0.4,
                    color=bottom_training_monitor.train_train_regions_color,
                    zorder=0
                    )
            # Add colored background spans to the plot (test)
            for xmin, xmax in bottom_training_monitor.test_test_regions:
                ax1.axvspan(
                    xmin, xmax, alpha=0.4,
                    color=bottom_training_monitor.test_test_regions_color,
                    zorder=0
                    )
            # Plot model metrics
            ax1.plot(
                bottom_training_monitor.train_training_metric_history,
                label="adaptative_training_history",
                color='tab:green',
                zorder=2
                )
            # Set Y limit
            ax1.set_ylim([-0.5, 14.5])
        if model_index == 2:
            # Add colored background spans to the plot (train)
            for xmin, xmax in time_training_monitor.train_train_regions:
                ax1.axvspan(
                    xmin, xmax, alpha=0.4,
                    color=time_training_monitor.train_train_regions_color,
                    zorder=0
                    )
            # Add colored background spans to the plot (test)
            for xmin, xmax in time_training_monitor.test_test_regions:
                ax1.axvspan(
                    xmin, xmax, alpha=0.4,
                    color=time_training_monitor.test_test_regions_color,
                    zorder=0
                    )
            # Plot model metrics
            ax1.plot(
                time_training_monitor.train_training_metric_history,
                label="adaptative_training_history",
                color='tab:green',
                zorder=2
                )
            # Set Y limit
            ax1.set_ylim([-0.5, 60.5])

        # Set Y label, grid and legend
        ax1.set_ylabel("% error", color='k')
        ax1.tick_params(axis='y', labelcolor='k')
        ax1.grid(True)
        ax1.legend()
        plt.tight_layout()  # Adjust subplot spacing

        # Create directory if it does not exit
        model_error_figures_dir = "./model_error_figures"
        if not os.path.exists(model_error_figures_dir):
            os.makedirs(model_error_figures_dir)

        # # Ask the user for the figure name
        # figure_save_file_name = input("Give me name to save this figure with "
        #                                 f"(path:{model_error_figures_dir}/<name>.pkl): ")

        # # Save the figure
        # with open(f"{model_error_figures_dir}/{figure_save_file_name}.pkl", 'wb') as file:
        #     pickle.dump(fig, file)

        # Plot the figure
        plt.show()


def every_models_inference(online_models_path):

    #
    # Get/read the model (already trained)
    #
    with open(online_models_path, "rb") as file:
        online_models_obj = pickle.load(file)

    # Create thread safe lock
    lock = threading.Lock()

    # Initialize the models
    online_models = om.OnlineModels(
        board="ZCU",
        lock=lock,
        input_top_model=online_models_obj[0],
        input_bottom_model=online_models_obj[1],
        input_time_model=online_models_obj[2])

    # Generate figures
    # show_model_data(online_models_obj)

    #
    # Generate combinations of features ¿What about CPU usage (continuous shit)?
    #

    # Calculate combinations without reposition of 'primary' elements taken in 'sub'
    primary = 11
    sub = 11
    max_slots = 8
    sub_combs = list(itertools.combinations(list(range(primary)), sub))
    # print(f"All combinations: {len(sub_combs)}")

    # Calculate combinations with reposition of 'sub' elements taken in 'result'
    result_combs = []
    for comb in sub_combs:

        for slot in range(max_slots, 0, -1):
            result_combs.extend(list(itertools.combinations_with_replacement(comb, slot)))
    # print(f"Slice combinations: {len(result_combs)}")

    # Remove duplications
    clean_result_combs = list(set(result_combs))
    print(f"Result combinations: {len(clean_result_combs)}")

    #
    # Infer each combination of features
    #

    # Construct features
    kernels_names = ["aes", "bulk", "crs", "kmp", "knn", "merge", "nw", "queue", "stencil2d", "stencil3d", "strided"]
    cpu_usage_names = ["user", "kernel", "idle"]
    predictions_names = ["top_power", "bottom_power", "time"]
    predictions_map = pd.DataFrame(columns=cpu_usage_names + kernels_names + predictions_names)
    num_iters = len(clean_result_combs)
    for i, comb in enumerate(clean_result_combs):

        # Show progress bar
        loop_progress_bar(i, num_iters, 0, True)

        # Count number of occurrences of each list item
        kernel_counts = Counter(comb)
        # print(kernel_counts)
        # Create feature dictionary
        features = {}
        # Set CPU usage static
        features["user"] = 0.2
        features["kernel"] = 0.2
        features["idle"] = 0.6
        # Get number of accs per kernel
        num_slots = 0
        for i, k in enumerate(kernels_names):
            # print(f"Instances of {k}: {kernel_counts[i]}")
            features[k] = kernel_counts[i]
            num_slots += kernel_counts[i]
        # print(features)

        # Predict (ZCU)
        top_power_prediction, \
            bottom_power_prediction, \
            time_prediction = \
            online_models.predict_one(features)

        # Save predicitons
        features["top_power"] = top_power_prediction
        features["bottom_power"] = bottom_power_prediction
        features["time"] = time_prediction
        features["num_slots"] = num_slots
        # features["top_power"] = 8
        # features["bottom_power"] = 9
        # features["time"] = 10

        # Append predictions to the map
        predictions_map = pd.concat([predictions_map, pd.DataFrame([features])], ignore_index=True)

    print(predictions_map)

    #
    # Save the "structure" in a file, so it doesn't have to be generated again
    #
    with open(f"./model_error_figures/all_predictions.pkl", 'wb') as file:
        pickle.dump(predictions_map, file)


# Repeat the predictions of the table of duration to see if they correlate
def generate_comparison_data(predictions_path):

    #
    # Get/read the predictions
    #
    with open(predictions_path, "rb") as file:
        predictions_df = pickle.load(file)

    # Print predictions with less than 8 slots
    print(predictions_df[predictions_df["num_slots"] < 8])

    # Repeat the figure

if __name__ == "__main__":

    # Models path
    online_models_path = "/home/juan/Documentos/doctorado/experiments/poisson_dist/src/run-time_processing/model_error_figures/adapt_models.pkl"
    predictions_path = "/home/juan/Documentos/doctorado/experiments/poisson_dist/src/run-time_processing/model_error_figures/all_predictions.pkl"
    #every_models_inference(online_models_path)
    generate_comparison_data(predictions_path)
