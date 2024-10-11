#!/usr/bin/env python3

"""
Visualize grid search information from a JSON file

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : October 2023
Description : This script reads a JSON file containing information about a grid
              search process and plots the error metrics and percentage of
              trained obsevations of the top, bottom and time models when
              varying certain parameters defined by the user in a configuration
              file "config/config.json".
"""
from cProfile import label
import json
import argparse
import sys
from pprint import pprint
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplcursors
import numpy as np


# Configuration file path
config_file_path = "config/config.json"
# Parameters file path
parameters_file_path = "config/parameters.json"

# Create the argument parser
parser = argparse.ArgumentParser()

# Indicate the path to the input JSON file
parser.add_argument('-i', dest="input_path", help='<Required> Path to the input JSON file', required=True)

# Indicate if the user want to see the pareto frontier instead
parser.add_argument('--pareto', action='store_true', help='Plot the Pareto-optimal points for the specified parameters configuration')

# Get the parsed arguments
args = parser.parse_args(sys.argv[1:])

# Open the JSON grid search file
with open(args.input_path, "r") as json_file:
    # Load the JSON data into a Python dictionary
    grid_search = json.load(json_file)

# Open the JSON file containing all the grid search parameters
# This is the parameters file format:
# {
#     "train": [
#         "nominal_obs_btw_validation",
#         "obs_btw_validation_reduction_factor",
#         "validation_threshold",
#         "stable_training_error_threshold"
#     ],
#     "test": [
#         "obs_btw_test",
#         "nominal_obs_to_test",
#         "test_threshold",
#         "obs_to_test_reduction_factor",
#         "significant_error_variation_threshold"
#     ]
# }
with open(parameters_file_path, "r") as json_file:
    parameters = json.load(json_file)

# Open the JSON configuration file
# This is the configuration file format: (some parameters are missing, meaning their value varies in the grid search)
# {
#     "parameters": {
#         "train": {
#             "nominal_obs_btw_validation": 1000,
#             "obs_btw_validation_reduction_factor": 0,
#             "stable_training_error_threshold": 2
#         },
#         "test": {
#             "obs_btw_test": 1000,
#             "nominal_obs_to_test": 200,
#             "obs_to_test_reduction_factor": 0,
#             "significant_error_variation_threshold": 2
#         }
#     }
# }
with open(config_file_path, "r") as json_file:
    configuration = json.load(json_file)

# Find which parameters are not set in the configuration file
# meaning that those are the ones we want to inspect in the grid search
variable_params = []
for group_of_params in parameters:
    for param in parameters[group_of_params]:
        if param not in configuration["parameters"][group_of_params]:
            variable_params.append((group_of_params, param))

# Print info about the variable parameters in this grid search
print("Variable Parameters ({}):".format(len(variable_params)))
for i, param in enumerate(variable_params):
    print("\t#{}: [{:^5}] -> {}".format(i, param[0].capitalize(), param[1]))

print("\nFixed Parameters:")
pprint(configuration["parameters"], sort_dicts=False)
print("")

# Create a dictionary to store information about the grid search
data = {}
data["parameter_labels"] = []
data["parameter_values"] = []
data["iteration"] = []
data["top_model"] = {}
data["top_model"]["error_values"] = []
data["top_model"]["trained_obs"] = []
data["top_model"]["trained_obs_percentage"] = []
data["bottom_model"] = {}
data["bottom_model"]["error_values"] = []
data["bottom_model"]["trained_obs"] = []
data["bottom_model"]["trained_obs_percentage"] = []
data["time_model"] = {}
data["time_model"]["error_values"] = []
data["time_model"]["trained_obs"] = []
data["time_model"]["trained_obs_percentage"] = []
for variable_param in variable_params:
    data["parameter_labels"].append((variable_param[0], variable_param[1]))
    data["parameter_values"].append([])

# pprint(data, sort_dicts=False)

# If the JSON file is a merge from other JSONs include this information
if "number_of_files" in grid_search:
    data["merge_info"] = {}
    data["merge_info"]["number_of_files"] = grid_search["number_of_files"]
    data["merge_info"]["files_info"] = grid_search["files_info"]
    # Flag for later use
    file_is_merged = True
else:
    file_is_merged = False

# Find all the iterations that share the same configuration
for iteration in grid_search["iterations"]:

    # This checks if the configuration dict is contained in the other
    # Only returns true is all the key:value pairs of the configuration dict are exactly contained in the other
    if configuration["parameters"]["train"].items() <= grid_search["iterations"][iteration]["parameters"]["train"].items() and configuration["parameters"]["test"].items() <= grid_search["iterations"][iteration]["parameters"]["test"].items():

        # Iterate over each variable parameter to store its corresponding value for this particular iteration
        for i, param in enumerate(data["parameter_labels"]):
            data["parameter_values"][i].append(grid_search["iterations"][iteration]["parameters"][param[0]][param[1]])

        # print("Top:\n\tMean Error: {} | Trained_obs: {}\nBottom:\n\tMean Error: {} | Trained_obs: {}\nTime:\n\tMean Error: {} | Trained_obs: {}\n".format(
        #     grid_search["iterations"][iteration]["models"]["top_model"]["adaptative"]["average_mape"],
        #     grid_search["iterations"][iteration]["models"]["top_model"]["adaptative"]["trained_observations"],
        #     grid_search["iterations"][iteration]["models"]["bottom_model"]["adaptative"]["average_mape"],
        #     grid_search["iterations"][iteration]["models"]["bottom_model"]["adaptative"]["trained_observations"],
        #     grid_search["iterations"][iteration]["models"]["time_model"]["adaptative"]["average_mape"],
        #     grid_search["iterations"][iteration]["models"]["time_model"]["adaptative"]["trained_observations"]
        # ))

        ################
        ## Store Data ##
        ################

        # Actual iteration
        data["iteration"].append(int(iteration))
        # Error and trained obs (for trained obs the percentage of them is also computed)
        data["top_model"]["error_values"].append(grid_search["iterations"][iteration]["models"]["top_model"]["adaptative"]["average_mape"])
        data["top_model"]["trained_obs"].append(grid_search["iterations"][iteration]["models"]["top_model"]["adaptative"]["trained_observations"])
        data["top_model"]["trained_obs_percentage"].append(round(grid_search["iterations"][iteration]["models"]["top_model"]["adaptative"]["trained_observations"] / grid_search["iterations"][iteration]["models"]["top_model"]["continuous"]["training_intervals"] * 100, 2))
        data["bottom_model"]["error_values"].append(grid_search["iterations"][iteration]["models"]["bottom_model"]["adaptative"]["average_mape"])
        data["bottom_model"]["trained_obs"].append(grid_search["iterations"][iteration]["models"]["bottom_model"]["adaptative"]["trained_observations"])
        data["bottom_model"]["trained_obs_percentage"].append(round(grid_search["iterations"][iteration]["models"]["bottom_model"]["adaptative"]["trained_observations"] / grid_search["iterations"][iteration]["models"]["bottom_model"]["continuous"]["training_intervals"] * 100, 2))
        data["time_model"]["error_values"].append(grid_search["iterations"][iteration]["models"]["time_model"]["adaptative"]["average_mape"])
        data["time_model"]["trained_obs"].append(grid_search["iterations"][iteration]["models"]["time_model"]["adaptative"]["trained_observations"])
        data["time_model"]["trained_obs_percentage"].append(round(grid_search["iterations"][iteration]["models"]["time_model"]["adaptative"]["trained_observations"] / grid_search["iterations"][iteration]["models"]["time_model"]["continuous"]["training_intervals"] * 100, 2))

# pprint(data, sort_dicts=False)

# If the user just want to see the Pareto this can be done for multiple variable parameters
# Otherwise just 2 could be display, so some comprobations need to be performced
if args.pareto:

    # Matplotlib configuration
    mpl.rcParams['figure.figsize'] = (15, 8)
    # Remove top and right frame
    mpl.rcParams['axes.spines.left'] = True
    mpl.rcParams['axes.spines.right'] = True
    mpl.rcParams['axes.spines.top'] = True
    mpl.rcParams['axes.spines.bottom'] = True

    # Define the model types
    model_types = ["top_model", "bottom_model", "time_model"]

    # User help info for possible actions
    print("Actions on red dots (Pareto Frontier):")
    print("\t(mouse) hover: Display iteration information as an annotation.")
    print("\t(mouse) left Click: print iteration information in terminal.")
    print("\t(mouse) right Click (on annotation): removes previously generated annotation.")
    print("\t(keyboard) 'v': toggles annotation's visibility.\n")

    # Compute iteration ranges if the JSON is a merge JSON
    # Generate local iteration info
    if file_is_merged:

        # Generate iteration ranges per file
        merged_files_iteration_ranges = []
        for file in range(data["merge_info"]["number_of_files"]):
            merged_files_iteration_ranges.append((data["merge_info"]["files_info"][str(file)]["first_iteration_position"], data["merge_info"]["files_info"][str(file)]["last_iteration_position"]))

        # Iterate over each iteration
        merged_files_iteration_info = []
        for iteration in data["iteration"]:
            # Store the iteration as global positin
            global_iteration = iteration
            # Find in which range of local iterations falls the global iteration and store the local file and iteration indexes
            for i, (start, end) in enumerate(merged_files_iteration_ranges):
                if start <= global_iteration <= end:
                    local_file = i
                    local_iteration = global_iteration - start
            # Append global iteration, local file and local iteration for each iteration
            merged_files_iteration_info.append((global_iteration, local_file, local_iteration))

    # Iterate over each model type
    for model in model_types:

        # Print model type in terminal
        print("{}\n".format(model.replace('_', ' ').capitalize()))

        # Generate a list of points that will conform the pareto points
        # For each point in data we get its id (iteration), mape and %train
        pareto_points = []
        for i, iteration in enumerate(data["iteration"]):
            pareto_points.append({"iteration": iteration, "MAPE": data[model]["error_values"][i], "trained_percentage": data[model]["trained_obs_percentage"][i]})

        # Sort the points based on its mape and % train
        pareto_points.sort(key=lambda point: (point["MAPE"], point["trained_percentage"]))

        print(f"{len(pareto_points)} total of pareto points")

        # Now we find with points conform the Pareto-optimal points, aka. the pareto frontier
        pareto_frontier = [pareto_points[0]]  # The first point is Pareto optimal by default
        for point in pareto_points[1:]:
            is_pareto = True
            to_remove = []
            # Pareto analysis, one data point is considered better than another
            # if it is at least as good in all dimensions and strictly better
            # in at least one dimension.
            # This is called Pareto dominance.
            for pf_point in pareto_frontier:
                if point["MAPE"] >= pf_point["MAPE"] and point["trained_percentage"] >= pf_point["trained_percentage"]:
                    is_pareto = False
                    break
                if pf_point["MAPE"] >= point["MAPE"] and pf_point["trained_percentage"] >= point["trained_percentage"]:
                    to_remove.append(pf_point)
            if is_pareto:
                pareto_frontier = [p for p in pareto_frontier if p not in to_remove]
                pareto_frontier.append(point)

        # Store also the non pareto frontier points
        no_pareto_frontier = [point for point in pareto_points if point not in pareto_frontier]

        # Extract pareto frontier x and y values as well as ids for plotting
        pareto_frontier_x_values = [point["MAPE"] for point in pareto_frontier]
        pareto_frontier_y_values = [point["trained_percentage"] for point in pareto_frontier]
        pareto_frontier_ids = [point["iteration"] for point in pareto_frontier]  # Extract IDs as strings

        # Extract pareto frontier x and y values for plotting
        no_pareto_frontier_x_values = [point["MAPE"] for point in no_pareto_frontier]
        no_pareto_frontier_y_values = [point["trained_percentage"] for point in no_pareto_frontier]
        no_pareto_frontier_ids = [point["iteration"] for point in no_pareto_frontier]  # Extract IDs as strings

        # Generate the pareto

        ##########
        ## Plot ##
        ##########

        # Create a figure
        fig, axis = plt.subplots(constrained_layout=True)

        # Select the figure title
        if model == "top_model":
            plot_title = "Top Power Model"
        elif model == "bottom_model":
            plot_title = "Bottom Power Model"
        else:
            plot_title = "Time Model"

        # Plot the Pareto frontier
        no_pareto_scatter = axis.scatter(no_pareto_frontier_x_values, no_pareto_frontier_y_values, label='No Pareto Frontier', color='y', edgecolors='black', marker='o', s=20)#s=25)
        pareto_scatter = axis.scatter(pareto_frontier_x_values, pareto_frontier_y_values, label='Pareto Frontier', color='r', edgecolors='black', marker='s')

        #########################################
        ## Local functions for cursor acctions ##
        #########################################

        def get_iteration_info(sel_index):
            "Generate a string with the information of this particular iteration"
            # Get the iteration to which the user is pointing with the cursor
            iteration = pareto_frontier_ids[sel_index]

            # Get the index of that iteration within the data error and %train lists
            actual_index = data["iteration"].index(iteration)

            # Introduce the iteration value in the string
            if file_is_merged:
                resulting_string = "Iteration: {} ({} -> {})\n".format(merged_files_iteration_info[actual_index][0], merged_files_iteration_info[actual_index][1], merged_files_iteration_info[actual_index][2])
            else:
                resulting_string = "Iteration: {}\n".format(iteration)

            resulting_string += "----------------------------\n"

            # Concatenate each variable parameter
            for i, param in enumerate(variable_params):
                resulting_string += "[{:^5}] -> {}: {}\n".format(param[0].capitalize(), param[1], data["parameter_values"][i][actual_index])

            # Concatenate training error and % of trained observations
            resulting_string += "----------------------------\n"
            resulting_string += "MAPE: {:.3f} %\n".format(data[model]["error_values"][actual_index])
            resulting_string += "Trained Observations: {} % ({})".format(data[model]["trained_obs_percentage"][actual_index], data[model]["trained_obs"][actual_index])

            # Return the string
            return resulting_string

        def cursor_annotation(sel):
            """Generate an annotation with iteration information when hovering cursor on pareto-optimal points"""

            # Generate de annotation (calling the function that generates the iteration info)
            sel.annotation.set_text(get_iteration_info(sel.index))
            # Change annotation transparency
            sel.annotation.get_bbox_patch().set(alpha=1)

        def cursor_print(sel):
            """Printing on terminal iteration information when clicking cursor on pareto-optimal points"""

            # Print the iteration information
            print(get_iteration_info(sel.index))
            print("----------------------------\n")
            # Make annotation invisible
            sel.annotation.set(visible=False)

        #############
        ## Cursors ##
        #############

        # Enable hover so when the user hovers the mouse over a point the annotation is generated
        cursor1 = mplcursors.cursor(pareto_scatter, hover=True)
        cursor1.connect("add", cursor_annotation)

        # Disable hover so when the user hovers the mouse over a point nothing happens
        # But when clicking the info of that iteration is printed
        cursor2 = mplcursors.cursor(pareto_scatter, hover=False)
        cursor2.connect("add", cursor_print)

        axis.minorticks_on()
        #axes.grid(alpha=0.5, linestyle=':', which="both")
        # Or if you want different settings for the grids:
        axis.grid(which='minor', linestyle=':', alpha=0.2)
        axis.grid(which='major', linestyle=':', alpha=0.5)
        plt.title(plot_title)
        plt.xlabel("MAPE (%)", fontsize=20)
        plt.ylabel("Trained Percentage (%)", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        axis.legend()

        plt.show()
        # When plotting the Pareto only the pareto is plotted, then we exit
    exit()


# Check if the parameters mark by the user to be variable in the grid search
# (the user actually does not include them)
# If we find that one or more parameters marked as variable do not vary
# we inform the user so he can modify the config.json. Because if the user
# marks 2 variable parameters but one is not really varying, the surface plot
# will only show a line. So it would be best to remove that non-varying param
# having the program to actually plot a line.
non_varying_parameters_count = 0
non_varying_parameters_indexes = []
non_varying_parameters_values = []
for i, parameter in enumerate(data["parameter_values"]):
    unique_param_values = set(parameter)
    if len(unique_param_values) == 1:
        non_varying_parameters_count += 1
        non_varying_parameters_indexes.append(i)
        non_varying_parameters_values.append(list(unique_param_values)[0])
# Show the non-varying parameters to the user and stops execution
if non_varying_parameters_count:
    print("The following user-defined variable pparameters do not actually vary ({}):".format(non_varying_parameters_count))
    for i in range(non_varying_parameters_count):
        print("\t#{} | [{:^5}] -> {}: {}".format(i, data["parameter_labels"][non_varying_parameters_indexes[i]][0].capitalize(), data["parameter_labels"][non_varying_parameters_indexes[i]][1], non_varying_parameters_values[i]))
    print("\nYou should remove those parameters from the config.json file and re-run this script :(")
    exit()

# Get the numbre of variable parameters (defined by the user in the config.json file)
num_variable_parameters = len(data["parameter_labels"])

# If there are more than 2 variable parameters (or none) we cannot represent anything
# With 2 variable parameters we represent a 3D surface and with 1 a classical plot
if num_variable_parameters < 0 or num_variable_parameters > 2:
    print("There are {} variable parameters. More than 2 cannot be represented. :(")

elif len(data["parameter_labels"]) == 1:

    # Matplotlib configuration
    mpl.rcParams['figure.figsize'] = (20, 12)
    # Remove top and right frame
    mpl.rcParams['axes.spines.left'] = True
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.bottom'] = True

    # Define the model types
    model_types = ["top_model", "bottom_model", "time_model"]

    # Create a 2x2 grid of subplots within the same figure
    fig, axis = plt.subplots(nrows=3, ncols=1, sharex=True, constrained_layout=False)

    fig.supxlabel("Value")
    fig.suptitle("[{}] {}".format(data["parameter_labels"][0][0].capitalize(), data["parameter_labels"][0][1]), fontsize=16, fontweight='bold')

    # Variables used for ploting the models
    ax = []

    # Iterate over each model type
    for i, model in enumerate(model_types):

        # Prepare the list of axes to introduce new data
        ax.append([[], []])

        ##########
        ## Plot ##
        ##########

        # Get axis for train error
        ax[i][0] = axis[i]
        # Set tiple
        ax[i][0].set_title("{}".format(model.replace('_', ' ').capitalize()), fontsize=14)
        # Plot model metrics
        ax[i][0].plot(data["parameter_values"][0], data[model]["error_values"], label="MAPE", color='tab:blue', marker='x')
        ax[i][0].set_ylabel("MAPE (%)", color='tab:blue')
        ax[i][0].tick_params(axis='y', labelcolor='tab:blue')
        # Grid
        ax[i][0].grid(which='major', color='#DDDDDD', linewidth=0.8)
        ax[i][0].grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
        ax[i][0].minorticks_on()

        # Get axis for trained obs
        ax[i][1] = ax[i][0].twinx()
        # Plot model metrics
        ax[i][1].plot(data["parameter_values"][0], data[model]["trained_obs_percentage"], label="Trained Observations", color='tab:orange', marker='x')
        ax[i][1].set_ylabel("Trained Observations (%)", color='tab:orange')
        ax[i][1].tick_params(axis='y', labelcolor='tab:orange')
        # Grid
        ax[i][1].grid(which='major', color='#DDDDDD', linewidth=0.8)
        ax[i][1].grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
        ax[i][1].minorticks_on()

        # Calculate the minimum and maximum values of both datasets
        error_range = (min(data[model]["error_values"]), max(data[model]["error_values"]))
        error_range_diff = error_range[1] - error_range[0]

        trained_obs_range = (min(data[model]["trained_obs_percentage"]), max(data[model]["trained_obs_percentage"]))
        trained_obs_range_diff = trained_obs_range[1] - trained_obs_range[0]

        # Compute the ylim values so the error occupies the top half of the plot and the trained obs the bottom half
        ax0_min_ylim = error_range[0] - 1.2 * error_range_diff                # We want it to be the top half, so the min value is its min value + the whole range (+20% so we leave a space between plots)
        ax0_max_ylim = error_range[1] + 0.05 * error_range_diff               # We want it to be the top half, so the max value is its max value + 5% its range to leave a space at the top
        ax1_min_ylim = trained_obs_range[0] - 0.05 * trained_obs_range_diff   # We want it to be the bottom half, so the min value is its min value + 5% its range to leave a space at the bottom
        ax1_max_ylim = trained_obs_range[1] + 1.2 * trained_obs_range_diff    # We want it to be the bottom half, so the max value is its max value + the whole range (+20% so we leave a space between plots)

        # Set the Y limits
        ax[i][0].set_ylim([ax0_min_ylim, ax0_max_ylim])
        ax[i][1].set_ylim([ax1_min_ylim, ax1_max_ylim])

        ############
        ## Legend ##
        ############

        # Get the lines and labels for each axis
        tmp_lines_0, tmp_labels_0 = ax[i][0].get_legend_handles_labels()
        tmp_lines_1, tmp_labels_1 = ax[i][1].get_legend_handles_labels()
        # Concatenate each axis labels and lines
        lines = tmp_lines_0 + tmp_lines_1
        labels = tmp_labels_0 + tmp_labels_1
        # Generate the legend
        ax[i][0].legend(lines, labels, loc="best")

        # plt.tight_layout()  # Adjust subplot spacing
        # plt.show()

# Plot a surface
elif num_variable_parameters == 2:

    # Matplotlib configuration
    mpl.rcParams['figure.figsize'] = (20, 12)
    # Remove top and right frame
    mpl.rcParams['axes.spines.left'] = True
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.bottom'] = True

    # Define the model types
    model_types = ["top_model", "bottom_model", "time_model"]

    # Iterate over each model type
    for model in model_types:

        # Create the figure
        fig = plt.figure(constrained_layout=True)

        # Select the figure title
        if model == "top_model":
            fig.suptitle("Top Power Model", fontsize=16, fontweight='bold')
        elif model == "bottom_model":
            fig.suptitle("Bottom Power Model", fontsize=16, fontweight='bold')
        else:
            fig.suptitle("Time Model", fontsize=16, fontweight='bold')

        # This create a grid with the following indexes
        #
        # grid_spec -> [rows (list), cols (list)]
        #
        # e.g.: grid_spec[1:, 1:] (desde la segunda fila hasta el final) (desde la segunda columna hasta el final)
        # 0 0 0
        # 0 x x
        # 0 x x
        grid_spec = fig.add_gridspec(2, 2)

        ##################
        ## Scatter plot ##
        ##################

        # Generate the subplot for the complete first column
        ax0 = fig.add_subplot(grid_spec[:, 0], projection='3d')

        # Create the scatter plot
        sc = ax0.scatter(data["parameter_values"][0], data["parameter_values"][1], data[model]["error_values"], c=data[model]["trained_obs_percentage"], cmap="plasma")

        # Apply a color map for the trained observations (4th dimension)
        fig.colorbar(sc, label='Trained Observations (%)')

        # Set the x and y labels
        ax0.set_xlabel("[{}] {}".format(data["parameter_labels"][0][0].capitalize(), data["parameter_labels"][0][1].capitalize()))
        ax0.set_ylabel("[{}] {}".format(data["parameter_labels"][1][0].capitalize(), data["parameter_labels"][1][1].capitalize()))
        ax0.set_zlabel("MAPE (%)")
        ax0.tick_params(axis='y')

        ###################
        ## Surface Plots ##
        ###################

        # Compute the generate a matrix of values from the parameters lists
        #
        # We need to do this conversion
        #
        # old: [1, 1, 1, 2, 2, 2, 3, 3, 3]
        #
        # new: [[1, 1, 1]
        #       [2, 2, 2]
        #       [3, 3, 3]]
        #
        # To do so, we extract the unique values on the list to another list
        # and get the lenth of that list. That is the shape of the resulting matrix

        # Convert the list to a set to get unique values
        unique_param_0_values = set(data["parameter_values"][0])
        unique_param_1_values = set(data["parameter_values"][1])

        # Find the number of different values, aka: the shape of the matrix
        param_shape = (len(unique_param_0_values), len(unique_param_1_values))

        # Generate the matrixes: reshape the parameter values arrays to match the grid dimensions
        param_0_grid = np.reshape(data["parameter_values"][0], param_shape)
        param_1_grid = np.reshape(data["parameter_values"][1], param_shape)

        # Reshape the errors and trained obs to match the grid dimensions
        error_values = np.reshape(data[model]["error_values"], param_shape)
        trained_obs_values = np.reshape(data[model]["trained_obs_percentage"], param_shape)

        # Generate the subplot for the complete top right plot
        ax1 = fig.add_subplot(grid_spec[0, 1], projection='3d')
        # Generate the 3D surface
        ax1.plot_surface(param_0_grid, param_1_grid, error_values, cmap='viridis', label='a')
        # Set the x, y and z labels
        ax1.set_xlabel("[{}] {}".format(data["parameter_labels"][0][0].capitalize(), data["parameter_labels"][0][1].capitalize()))
        ax1.set_ylabel("[{}] {}".format(data["parameter_labels"][1][0].capitalize(), data["parameter_labels"][1][1].capitalize()))
        ax1.set_zlabel("MAPE (%)")
        ax1.tick_params(axis='y')

        # Generate the subplot for the complete top bottom plot
        ax2 = fig.add_subplot(grid_spec[1, 1], projection='3d')
        # Generate the 3D surface
        ax2.plot_surface(param_0_grid, param_1_grid, trained_obs_values, cmap='viridis', label='b')
        # Set the x, y and z labels
        ax2.set_xlabel("[{}] {}".format(data["parameter_labels"][0][0].capitalize(), data["parameter_labels"][0][1].capitalize()))
        ax2.set_ylabel("[{}] {}".format(data["parameter_labels"][1][0].capitalize(), data["parameter_labels"][1][1].capitalize()))
        ax2.set_zlabel("Trained obsevarions (%)")
        ax2.tick_params(axis='y')

        ####################
        ## Multiple plots ##
        ####################

        # Create a 2X1 grid of subplots within the same figure
        fig2, axis2 = plt.subplots(nrows=2, ncols=1, sharex=True, constrained_layout=True)

        # Add a common X label
        fig2.supxlabel("[{}] {}".format(data["parameter_labels"][1][0].capitalize(), data["parameter_labels"][1][1].capitalize()))

        # Select the figure title
        if model == "top_model":
            fig2.suptitle("Top Power Model", fontsize=16, fontweight='bold')
        elif model == "bottom_model":
            fig2.suptitle("Bottom Power Model", fontsize=16, fontweight='bold')
        else:
            fig2.suptitle("Time Model", fontsize=16, fontweight='bold')

        # Error plot

        # Get the axis of the first plot in the grid
        axis2_0 = axis2[0]
        # Set title
        # axis2_0.set_title("Error", fontsize=14)
        # Plot model metrics
        for i, param_0_list in enumerate(param_0_grid):
            axis2_0.plot(param_1_grid[i], error_values[i], label="[{}] {} = {}".format(data["parameter_labels"][0][0].capitalize(), data["parameter_labels"][0][1].capitalize(), param_0_list[0]), marker='o')
        axis2_0.set_ylabel("MAPE (%)")
        axis2_0.tick_params(axis='y')
        # Grid
        axis2_0.grid(which='major', color='#DDDDDD', linewidth=0.8)
        axis2_0.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
        axis2_0.minorticks_on()
        axis2_0.legend(loc="best")

        # Trained percentage plot

        # Get the axis of the first plot in the grid
        axis2_1 = axis2[1]
        # Set title
        # axis2_1.set_title("Train %", fontsize=14)
        # Plot model metrics
        for i, param_0_list in enumerate(param_0_grid):
            axis2_1.plot(param_1_grid[i], trained_obs_values[i], label="[{}] {} = {}".format(data["parameter_labels"][0][0].capitalize(), data["parameter_labels"][0][1].capitalize(), param_0_list[0]), marker='o')
        axis2_1.set_ylabel("Trained obsevarions (%)")
        axis2_1.tick_params(axis='y')
        # Grid
        axis2_1.grid(which='major', color='#DDDDDD', linewidth=0.8)
        axis2_1.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
        axis2_1.minorticks_on()
        axis2_1.legend(loc="best")


# Prints for the user
# pprint(data)

# Check if the JSON is a merged JSON
if file_is_merged:
    print("This JSON file is a merge of other {} JSON files.\n".format(data["merge_info"]["number_of_files"]))

    for file in range(data["merge_info"]["number_of_files"]):
        print("File #{} info:".format(file))
        pprint(data["merge_info"]["files_info"][str(file)])
else:
    print("This JSON file is not a merge of other JSON files.")

# Extract the parameter groups (train, test, etc)
parameter_groups = [parameter[0] for parameter in data["parameter_labels"]]

# Get the max lenght of the parameter groups, for formating
# max(list, key:len) returns the string of highest len on the list
# Obtaining its lenght gives us the parameter groups max lenght
parameter_group_max_lenght = len(max(parameter_groups, key=len))

# Extract all the parameters (from all the parameter groups)
parameters_list = [parameter[1] for parameter in data["parameter_labels"]]

# Get the max lenght of the parameter, for formating
# max(list, key:len) returns the string of highest len on the list
# Obtaining its lenght gives us the parameter max lenght
parameters_max_lenght = len(max(parameters_list, key=len))

# Extract all the parameters values (from all the parameter groups)
parameters_values = [parameter for parameter_list in data["parameter_values"] for parameter in parameter_list]

# Get the max lenght of the parameter, for formating
# max(list, key:len) returns the string of highest len on the list
# Obtaining its lenght gives us the parameter max lenght
parameters_values_as_strings = [str(value) for value in parameters_values]
parameters_values_max_lenght = len(max(parameters_values_as_strings, key=len))
# Check if the header is larger than the largest iteration value
parameters_values_max_lenght = max(len("Value"), parameters_values_max_lenght)

# Print table with information
if file_is_merged:

    # Generate iteration ranges per file
    merged_files_iteration_ranges = []
    for file in range(data["merge_info"]["number_of_files"]):
        merged_files_iteration_ranges.append((data["merge_info"]["files_info"][str(file)]["first_iteration_position"], data["merge_info"]["files_info"][str(file)]["last_iteration_position"]))

    # Get the max lenght of the iterations, for formating
    # max(list, key:len) returns the string of highest len on the list
    # Obtaining its lenght gives us the iteration max lenght
    # (a string is created containing the global iteration value, as well as the local file index and the local position of the globar iteration within that file)
    iterations_as_strings = []
    # Iterate over each iteration
    for iteration in data["iteration"]:
        # Store the iteration as global positin
        global_iteration = iteration
        # Find in which range of local iterations falls the global iteration and store the local file and iteration indexes
        for i, (start, end) in enumerate(merged_files_iteration_ranges):
            if start <= global_iteration <= end:
                local_file = i
                local_iteration = global_iteration - start
        # Create a string with that information
        iterations_as_strings.append("{} ({}->{})".format(global_iteration, local_file, local_iteration))
    iteration_max_lenght = len(max(iterations_as_strings, key=len))
    # Check if the header is larger than the largest iteration value
    iteration_max_lenght = max(len("Iteration (lfile->iter)"), iteration_max_lenght)

    # Generate the table separator string
    table_separator = "+-{}-+-{}---{}-+-{}-+".format(
        "-" * iteration_max_lenght,
        "-" * parameter_group_max_lenght,
        "-" * parameters_max_lenght,
        "-" * parameters_values_max_lenght,)

    # Print the header
    print("")
    print(table_separator)
    print("| {:^{i}} | {:>{g}} - {:<{p}} | {:^{v}} |".format(
        "Iteration (lfile->iter)",
        "Group",
        "Parameter",
        "Value",
        i=iteration_max_lenght,
        g=parameter_group_max_lenght,
        p=parameters_max_lenght,
        v=parameters_values_max_lenght))

    # Print each iteration information
    for i in range(len(data["iteration"])):
        print(table_separator)
        for j, param in enumerate(data["parameter_labels"]):
            if j == 0:
                print("| {:^{s}} | {:<{g}} - {:<{p}} | {:>{v}} |".format(
                    iterations_as_strings[i],
                    param[0],
                    param[1],
                    data["parameter_values"][j][i],
                    s=iteration_max_lenght,
                    g=parameter_group_max_lenght,
                    p=parameters_max_lenght,
                    v=parameters_values_max_lenght))
            else:
                print("| {:^{s}} | {:<{g}} - {:<{p}} | {:>{v}} |".format(
                    "",
                    param[0],
                    param[1],
                    data["parameter_values"][j][i],
                    s=iteration_max_lenght,
                    g=parameter_group_max_lenght,
                    p=parameters_max_lenght,
                    v=parameters_values_max_lenght))
    print(table_separator)

else:

    # Get the max lenght of the iterations, for formating
    # max(list, key:len) returns the string of highest len on the list
    # Obtaining its lenght gives us the iteration max lenght
    iterations_as_strings = [str(iteration) for iteration in data["iteration"]]
    iteration_max_lenght = len(max(iterations_as_strings, key=len))
    # Check if the header is larger than the largest iteration value
    iteration_max_lenght = max(len("Iteration"), iteration_max_lenght)

    # Generate the table separator string
    table_separator = "+-{}-+-{}---{}-+-{}-+".format(
        "-" * iteration_max_lenght,
        "-" * parameter_group_max_lenght,
        "-" * parameters_max_lenght,
        "-" * parameters_values_max_lenght,)

    # Print the header
    print("")
    print(table_separator)
    print("| {:^{i}} | {:>{g}} - {:<{p}} | {:^{v}} |".format(
        "Iteration",
        "Group",
        "Parameter",
        "Value",
        i=iteration_max_lenght,
        g=parameter_group_max_lenght,
        p=parameters_max_lenght,
        v=parameters_values_max_lenght))

    # Print each iteration information
    for i in range(len(data["iteration"])):
        print(table_separator)
        for j, param in enumerate(data["parameter_labels"]):
            if j == 0:
                print("| {:^{s}} | {:<{g}} - {:<{p}} | {:>{v}} |".format(
                    data["iteration"][i],
                    param[0],
                    param[1],
                    data["parameter_values"][j][i],
                    s=iteration_max_lenght,
                    g=parameter_group_max_lenght,
                    p=parameters_max_lenght,
                    v=parameters_values_max_lenght))
            else:
                print("| {:^{s}} | {:<{g}} - {:<{p}} | {:>{v}} |".format(
                    "",
                    param[0],
                    param[1],
                    data["parameter_values"][j][i],
                    s=iteration_max_lenght,
                    g=parameter_group_max_lenght,
                    p=parameters_max_lenght,
                    v=parameters_values_max_lenght))
    print(table_separator)

# Show plots only when generated (num_variable_parameters under 2)
if 0 < num_variable_parameters <= 2:
    # Show the plots
    plt.show()

# TODO: Si selecionas más de un parametro como variable pero luego el json no contiene multiples valores para alguno de ellos
#       Te va a pasar que vas a dibujar una superficie pero realmente será una linea, porque en uno de los dos ejes el
#       valor del parámetro será siempre el mismo... Es problema del usuario asegurarse de elegir adecuadamente los parámetros
