#!/usr/bin/env python3

"""
Merge JSON files containing grid search processes information

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : October 2023
Description : This script merges multiple JSON files containing the information
              of different grid search processes.
"""

import json
import argparse
import sys

# Create the argument parser
parser = argparse.ArgumentParser()

# Indicate the path of the files containing the JSONs.
parser.add_argument('-p', dest="json_files_paths", nargs='+', help='<Required> Paths to the JSON files to merge', required=True)

# Indicate the path to which the merged JSON must be written.
parser.add_argument('-o', dest="output_path", help='<Required> Output JSON path (including the file name)', required=True)

# Indicate if the user want to remove the training regions information (too wordy).
parser.add_argument('--remove_training_regions', action='store_true', help='Remove training regions info')

# Get the parsed arguments
args = parser.parse_args(sys.argv[1:])

json_file_content_list = []
# Loop over each JSON path
for json_file_path in args.json_files_paths:

    # Open the JSON file
    with open(json_file_path, "r") as json_file:
        # Load the JSON data into a Python dictionary
        json_file_content_list.append(json.load(json_file))

# Generate a dictionary with the merging information
merged_json = {}

# Number of files
merged_json["number_of_files"] = len(json_file_content_list)

#############
## Merging ##
#############

merged_json["files_info"] = {}
merged_json["iterations"] = {}
already_proccessed_iterations = 0

# List that will contain the training errors for subsequent re-sorting
top_error_list = []
top_train_obs_list = []
bottom_error_list = []
bottom_train_obs_list = []
time_error_list = []
time_train_obs_list = []

for i, json_content in enumerate(json_file_content_list):

    # Create a dictionary per file
    merged_json["files_info"][str(i)] = {}
    # Get the JSON file path
    merged_json["files_info"][str(i)]["path"] = args.json_files_paths[i]
    # Get the number of iterations per JSON file
    merged_json["files_info"][str(i)]["total_iterations"] = len(json_content["iterations"])
    # Set the position of the first iteration of the JSON file inside the merged JSON
    # e.g. if the first JSON has 200 iterations the first iteration of the
    # second JSON would be the 201th itertion (they are ordered consecutively)
    merged_json["files_info"][str(i)]["first_iteration_position"] = already_proccessed_iterations
    # Set the position of the last iteration of the JSON file inside the merged JSON
    merged_json["files_info"][str(i)]["last_iteration_position"] = already_proccessed_iterations + len(json_content["iterations"]) - 1

    # Re-generate the iteration value for the merged files
    for iteration in range(len(json_content["iterations"])):

        # Remove the training_regions if specified by the user
        if args.remove_training_regions:
            del json_content["iterations"][(str(iteration))]["models"]["top_model"]["adaptative"]["training_regions"]
            del json_content["iterations"][(str(iteration))]["models"]["bottom_model"]["adaptative"]["training_regions"]
            del json_content["iterations"][(str(iteration))]["models"]["time_model"]["adaptative"]["training_regions"]

        # Replace the key name keeping its value
        merged_json["iterations"][str(already_proccessed_iterations + iteration)] = json_content["iterations"][str(iteration)]
        # Store error metrics and trained observations for a posterior sorting
        top_error_list.append(json_content["iterations"][str(iteration)]["models"]["top_model"]["adaptative"]["average_mape"])
        top_train_obs_list.append(json_content["iterations"][str(iteration)]["models"]["top_model"]["adaptative"]["trained_observations"])
        bottom_error_list.append(json_content["iterations"][str(iteration)]["models"]["bottom_model"]["adaptative"]["average_mape"])
        bottom_train_obs_list.append(json_content["iterations"][str(iteration)]["models"]["bottom_model"]["adaptative"]["trained_observations"])
        time_error_list.append(json_content["iterations"][str(iteration)]["models"]["time_model"]["adaptative"]["average_mape"])
        time_train_obs_list.append(json_content["iterations"][str(iteration)]["models"]["time_model"]["adaptative"]["trained_observations"])

    # Increment the already proccessed iterations counter
    already_proccessed_iterations += len(json_content["iterations"])

# Sort the error and training obs and add the corresponding index to each iteration
# Use enumerate to get (index, value) pairs and sort them by value
top_error_sorted_indices = sorted(enumerate(top_error_list), key=lambda x: x[1])
bottom_error_sorted_indices = sorted(enumerate(bottom_error_list), key=lambda x: x[1])
time_error_sorted_indices = sorted(enumerate(time_error_list), key=lambda x: x[1])
top_train_obs_sorted_indices = sorted(enumerate(top_train_obs_list), key=lambda x: x[1])
bottom_train_obs_sorted_indices = sorted(enumerate(bottom_train_obs_list), key=lambda x: x[1])
time_train_obs_sorted_indices = sorted(enumerate(time_train_obs_list), key=lambda x: x[1])
# Extract the indices from the sorted list
top_error_sorted_indices = [index for index, value in top_error_sorted_indices]
bottom_error_sorted_indices = [index for index, value in bottom_error_sorted_indices]
time_error_sorted_indices = [index for index, value in time_error_sorted_indices]
top_train_obs_sorted_indices = [index for index, value in top_train_obs_sorted_indices]
bottom_train_obs_sorted_indices = [index for index, value in bottom_train_obs_sorted_indices]
time_train_obs_sorted_indices = [index for index, value in time_train_obs_sorted_indices]
# Add the sorted position (in term of best error and less train obs) to the corresponding models        
for position, (top_error_index, bottom_error_index, time_error_index, top_train_obs_index, bottom_train_obs_index, time_train_obs_index) in enumerate(zip(top_error_sorted_indices, bottom_error_sorted_indices, time_error_sorted_indices, top_train_obs_sorted_indices, bottom_train_obs_sorted_indices, time_train_obs_sorted_indices)):
    merged_json["iterations"][str(top_error_index)]["models"]["top_model"]["positions"]["best_error"] = position
    merged_json["iterations"][str(top_train_obs_index)]["models"]["top_model"]["positions"]["less_train"] = position
    merged_json["iterations"][str(bottom_error_index)]["models"]["bottom_model"]["positions"]["best_error"] = position
    merged_json["iterations"][str(bottom_train_obs_index)]["models"]["bottom_model"]["positions"]["less_train"] = position
    merged_json["iterations"][str(time_error_index)]["models"]["time_model"]["positions"]["best_error"] = position
    merged_json["iterations"][str(time_train_obs_index)]["models"]["time_model"]["positions"]["less_train"] = position

# Generate the best_models dictionary
merged_json["best_models"] = {
    "top": {
        "best_error": {
            "value": merged_json["iterations"][str(top_error_sorted_indices[0])]["models"]["top_model"]["adaptative"]["average_mape"],
            "index": top_error_sorted_indices[0]
        },
        "less_train": {
            "value": merged_json["iterations"][str(top_train_obs_sorted_indices[0])]["models"]["top_model"]["adaptative"]["trained_observations"],
            "index": top_train_obs_sorted_indices[0]
        }
    },
    "bottom": {
        "best_error": {
            "value": merged_json["iterations"][str(bottom_error_sorted_indices[0])]["models"]["bottom_model"]["adaptative"]["average_mape"],
            "index": bottom_error_sorted_indices[0]
        },
        "less_train": {
            "value": merged_json["iterations"][str(bottom_train_obs_sorted_indices[0])]["models"]["bottom_model"]["adaptative"]["trained_observations"],
            "index": bottom_train_obs_sorted_indices[0]
        }
    },
    "time": {
        "best_error": {
            "value": merged_json["iterations"][str(time_error_sorted_indices[0])]["models"]["time_model"]["adaptative"]["average_mape"],
            "index": time_error_sorted_indices[0]
        },
        "less_train": {
            "value": merged_json["iterations"][str(time_train_obs_sorted_indices[0])]["models"]["time_model"]["adaptative"]["trained_observations"],
            "index": time_train_obs_sorted_indices[0]
        }
    }
}

# Write the merged JSON
with open("{}.json".format(args.output_path), "w") as file:
    # Dump the dictionary to the file with indentation for pretty printing
    json.dump(merged_json, file, indent=4)

# Print ack to the user
print("\n{} JSON files merged succesfully.".format(merged_json["number_of_files"]))
print("\nThe resulting JSON has been stored in '{}.json'.\n".format(args.output_path))