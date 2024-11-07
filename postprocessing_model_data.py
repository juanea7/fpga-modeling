#!/usr/bin/env python3

"""
Post-processing Online Models Data

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : December 2023
Description : Processes the online models output data and generates error metrics figures

"""

import os
import sys
import pickle
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def show_model_data(online_models_data, board):
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
    models_list = []
    if board["power"]["rails"] == "dual":
        models_list.append(online_models_data[0])
        models_list.append(online_models_data[1])
        models_list.append(online_models_data[2])
    elif board["power"]["rails"] == "mono":
        models_list.append(online_models_data[0])
        models_list.append(online_models_data[1])
    else:
        raise ValueError("Board not supported: {}".format(board))

    # Matplotlib configuration
    mpl.rcParams['figure.figsize'] = (20, 12)
    # Remove top and right frame
    mpl.rcParams['axes.spines.left'] = True
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.bottom'] = True


    for iter, model in enumerate(models_list):
        # Create a 2x2 grid of subplots within the same figure
        fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True, constrained_layout=False)

        fig.supxlabel('Number of Observations')
        fig.suptitle('Error Metrics!!')

        # Add colored background spans to the plot (train)
        for xmin, xmax in model.train_train_regions:
            ax1.axvspan(
                xmin, xmax, alpha=0.4,
                color=model.train_train_regions_color,
                zorder=0
                )
        # Add colored background spans to the plot (test)
        for xmin, xmax in model.test_test_regions:
            ax1.axvspan(
                xmin, xmax, alpha=0.4,
                color=model.test_test_regions_color,
                zorder=0
                )
        # Plot model metrics
        ax1.plot(
            model.train_training_metric_history,
            label="adaptative_training_history",
            color='tab:green',
            zorder=2
            )
        # Set Y limit based on the number of models and their index
        if (len(models_list) == 3 and iter < 2) or (len(models_list) == 2 and iter < 1):
            ax1.set_ylim([-0.5, 14.5])
        else:
            ax1.set_ylim([-0.5, 60.5])

        print(f"Model {iter} - Average Training Error: {np.mean(model.train_training_metric_history)}")


        # Set Y label, grid and legend
        ax1.set_ylabel("% error", color='k')
        ax1.tick_params(axis='y', labelcolor='k')
        ax1.grid(True)
        ax1.legend()
        plt.tight_layout()  # Adjust subplot spacing

        # Create directory if it does not exit
        #model_error_figures_dir = "./model_error_figures"
        #if not os.path.exists(model_error_figures_dir):
        #    os.makedirs(model_error_figures_dir)

        # # Ask the user for the figure name
        # figure_save_file_name = input("Give me name to save this figure with "
        #                                 f"(path:{model_error_figures_dir}/<name>.pkl): ")

        # # Save the figure
        # with open(f"{model_error_figures_dir}/{figure_save_file_name}.pkl", 'wb') as file:
        #     pickle.dump(fig, file)

        # Plot the figure
        plt.show()


def show_model_data_with_time(online_models_data, times_list):
    """
    Process models output data and generate pertinent figures.

    Online Models Data: [
            online_models._top_power_model._training_monitor,
            online_models._bottom_power_model._training_monitor,
            online_models._time_model._training_monitor
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

        fig.supxlabel('Time (s)')
        fig.suptitle('Error Metrics!!')

        if model_index == 0:
            # Add colored background spans to the plot (train)
            for xmin, xmax in top_training_monitor.train_train_regions:
                # It could happen that the program stops while testing/traing, this generates
                # an xmin index without xmax and rises errores. Just omit them
                if xmin >= len(times_list) or xmax >= len(times_list): continue
                ax1.axvspan(
                    times_list[xmin], times_list[xmax], alpha=0.4,
                    color=top_training_monitor.train_train_regions_color,
                    zorder=0
                    )
            # Add colored background spans to the plot (test)
            for xmin, xmax in top_training_monitor.test_test_regions:
                # It could happen that the program stops while testing/traing, this generates
                # an xmin index without xmax and rises errores. Just omit them
                if xmin >= len(times_list) or xmax >= len(times_list): continue
                ax1.axvspan(
                    times_list[xmin], times_list[xmax], alpha=0.4,
                    color=top_training_monitor.test_test_regions_color,
                    zorder=0
                    )
            # Plot model metrics
            ax1.plot(
                times_list,
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
                # It could happen that the program stops while testing/traing, this generates
                # an xmin index without xmax and rises errores. Just omit them
                if xmin >= len(times_list) or xmax >= len(times_list): continue
                ax1.axvspan(
                    times_list[xmin], times_list[xmax], alpha=0.4,
                    color=bottom_training_monitor.train_train_regions_color,
                    zorder=0
                    )
            # Add colored background spans to the plot (test)
            for xmin, xmax in bottom_training_monitor.test_test_regions:
                # It could happen that the program stops while testing/traing, this generates
                # an xmin index without xmax and rises errores. Just omit them
                if xmin >= len(times_list) or xmax >= len(times_list): continue
                ax1.axvspan(
                    times_list[xmin], times_list[xmax], alpha=0.4,
                    color=bottom_training_monitor.test_test_regions_color,
                    zorder=0
                    )
            # Plot model metrics
            ax1.plot(
                times_list,
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
                # It could happen that the program stops while testing/traing, this generates
                # an xmin index without xmax and rises errores. Just omit them
                if xmin >= len(times_list) or xmax >= len(times_list): continue
                ax1.axvspan(
                    times_list[xmin], times_list[xmax], alpha=0.4,
                    color=time_training_monitor.train_train_regions_color,
                    zorder=0
                    )
            # Add colored background spans to the plot (test)
            for xmin, xmax in time_training_monitor.test_test_regions:
                # It could happen that the program stops while testing/traing, this generates
                # an xmin index without xmax and rises errores. Just omit them
                if xmin >= len(times_list) or xmax >= len(times_list): continue
                ax1.axvspan(
                    times_list[xmin], times_list[xmax], alpha=0.4,
                    color=time_training_monitor.test_test_regions_color,
                    zorder=0
                    )
            # Plot model metrics
            ax1.plot(
                times_list,
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


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()

    # Indicate the board
    parser.add_argument(
        '-b',
        dest="board",
        help='<Required> Board used for the training',
        choices=["ZCU", "PYNQ", "AU250"],
        required=True
        )

    # Indicate the path of the file containing the online models data
    parser.add_argument(
        '-m',
        dest="models_training_monitors_path",
        help='<Required> Path to the file containing the online models training monitor',
        required=True
        )

    # Indicate the path of the file containing the temporal data
    parser.add_argument(
        '-t',
        dest="temporal_data_path",
        default=None,
        help='<optional> Path to the file containing the temporal data'
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


    # Open models data
    with open(args.models_training_monitors_path, "rb") as file:
        online_models_obj = pickle.load(file)

    # Generate figures
    show_model_data(online_models_obj, board)

    if args.temporal_data_path is not None:

        # temporal data
        with open(args.temporal_data_path, "rb") as file:
            temporal_data = pickle.load(file)

        # TODO: Testing. Process temporal data
        #
        # Temporal Data: [
        #     num_measurements,
        #     start_time_train/test,
        #     start_observation_index,
        #     end_time_train/test,
        #     end_observation_index
        #     ]
        #

        observations_index_list = []
        observations_time_list = []

        MEASURING_PERIOD_S = 0.5  # 500 ms

        for i, [measurements, start_time, start_index, end_time, end_index] in enumerate(temporal_data):

            if i == 0:
                init_time = start_time - measurements * MEASURING_PERIOD_S
                start_operation_time = 0.0
            else:
                start_operation_time = start_time - init_time - measurements * MEASURING_PERIOD_S

            end_operation_time = end_time - init_time

            # Introduce data in the list
            observations_index_list.append(start_index)
            observations_time_list.append(start_operation_time)
            observations_index_list.append(end_index)
            observations_time_list.append(end_operation_time)

        # Linear interpolation function
        interp_func = interp1d(
            observations_index_list,
            observations_time_list,
            kind='linear',
            fill_value='extrapolate'
            )

        # Generate times for all indices in the magnitude list
        time_instants = interp_func(np.arange(len(online_models_obj[0].train_training_metric_history)))

        show_model_data_with_time(online_models_obj, time_instants)
