#!/usr/bin/env python3

"""
Processing of the observations

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : October 2022
Description : This contains the functions usefull for processing the
              observations that would later be fed to the models.

"""


import pandas as pd
from sklearn.model_selection import train_test_split


def read_observations_from_file(path_to_obs):
    """Read observations from contained in a file from a path."""

    # Read pickel file
    observations_df = pd.read_pickle(path_to_obs)

    # Test
    print("\nObservations:\n")
    print(observations_df)
    print("\n")

    return observations_df


def sample_observations(df):
    """Split a dataset in train and test dataset"""

    # Random sampling of train and test sets
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

    # Test
    print("\nTrain:\n")
    print(train_set)
    print("\n")

    print("\ntest:\n")
    print(test_set)
    print("\n")

    return train_set, test_set


def dataset_formating_zcu(dataframe):
    """Format a dataset extracting top power, bottom power and time features
       and labels
    """

    # Extract features
    features_df = dataframe.drop(["Top power", "Bottom power", "Time"], axis=1)

    # Extract labels
    top_power_labels_df = dataframe["Top power"].copy()
    bottom_power_labels_df = dataframe["Bottom power"].copy()
    time_labels_df = dataframe["Time"].copy()

    # Test
    # print("\nTop Power Attributes:\n")
    # print(top_power_labels_df)
    # print("\n")
    # print("\nBottom Power Attributes:\n")
    # print(bottom_power_labels_df)
    # print("\n")
    # print("\nTime Attributes:\n")
    # print(time_att_df)
    # print("\n")

    return [features_df, top_power_labels_df, bottom_power_labels_df, time_labels_df]


def dataset_formating_pynq(dataframe):
    """Format a dataset extracting power and time features
       and labels
    """

    # Extract features
    features_df = dataframe.drop(["Power", "Time"], axis=1)

    # Extract labels
    power_labels_df = dataframe["Power"].copy()
    time_labels_df = dataframe["Time"].copy()

    # Test
    # print("\nPower Attributes:\n")
    # print(power_labels_df)
    # print("\n")
    # print("\n")
    # print("\nTime Attributes:\n")
    # print(time_att_df)
    # print("\n")

    return [features_df, power_labels_df, time_labels_df]


# Map board to functions
formating_functions = {
    "ZCU": dataset_formating_zcu,
    "PYNQ": dataset_formating_pynq
    # TODO: Implement AU250
}


def dataset_formating(dataframe, board):
    """Format a dataset extracting power and time features
       and labels
    """

    if board not in formating_functions:
        raise ValueError(f"Board not supported: {board}")
    
    return formating_functions[board](dataframe)


if __name__ == "__main__":

    # Read observations file
    observations_dataframe = read_observations_from_file("./observations.pkl")

    # Sample train and test sets from the observations
    train_set_dataframe, \
        test_set_dataframe = sample_observations(observations_dataframe)

    # Format each dataset
    [att_train_dataframe, \
        top_power_labels_train_dataframe, \
        bottom_power_labels_train_dataframe, \
        time_labels_train_dataframe] = dataset_formating(train_set_dataframe, "ZCU")
