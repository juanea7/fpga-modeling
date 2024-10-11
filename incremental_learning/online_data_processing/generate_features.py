"""
Compute features from observations (returned in multiple formats)

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : October 2022
Description : Functions on this file compute features from given observations
              and return them either in a list or a dataframe.
"""

import pandas as pd


def process_observation_zcu(observation, cpu_usage: False):
    """
    Processes the observation to obtain the features to train the model
    Feature format: {
        Main_tag,
        aes_cu,
        bulk_cu,
        crs_cu,
        kmp_cu,
        knn_cu,
        merge_cu,
        nw_cu,
        queue_cu,
        stencil2d_cu,
        stencil3d_cu,
        strided_cu,
        top_power,
        bottom_power,
        time,
    }
    """

    # Name of each of the kernels
    benchmarks = [
        "aes",
        "bulk",
        "crs",
        "kmp",
        "knn",
        "merge",
        "nw",
        "queue",
        "stencil2d",
        "stencil3d",
        "strided"
    ]

    # Create a dict that will contain the processed features of the observation
    features = {}

    # If cpu usage, set the first features to that
    if cpu_usage:
        features["user"] = observation[0][0]
        features["kernel"] = observation[0][1]
        features["idle"] = observation[0][2]

    # Set the first feature, corresponding with the main kernel of the
    # observation
    features["Main"] = observation[-5]

    # Generate a local dict to store observation info
    observation_info = {}

    # Obtain a list with the info of all of the kernels in this observation
    kernels = observation[-4].split('_')

    # Iterate over each of the kernels in the observation processing its
    # particular features
    for kernel in kernels:
        # Split the information (name, cu, position of first cu *useless*)
        kernel_keywords = kernel.split('-')
        kernel_name = kernel_keywords[0]
        kernel_accelerators = kernel_keywords[1]
        # Asociate the cus to the kernel name in the temporal dict
        observation_info[kernel_name] = kernel_accelerators

    # Create the features of the observation, each kernel is represented by
    # a feature that indicates the number of cus of that particular kernel in
    # this observation (0 if none)
    for benchmark in benchmarks:
        features[benchmark] = int(observation_info.get(benchmark, 0))

    # Add the top/bottom power and time features of the observation
    features['Top power'] = observation[-3]
    features['Bottom power'] = observation[-2]
    features['Time'] = observation[-1]

    return features


def process_observation_pynq(observation, cpu_usage: False):
    """
    Processes the observation to obtain the features to train the model
    Feature format: {
        Main_tag,
        aes_cu,
        bulk_cu,
        crs_cu,
        kmp_cu,
        knn_cu,
        merge_cu,
        nw_cu,
        queue_cu,
        stencil2d_cu,
        stencil3d_cu,
        strided_cu,
        power,
        time,
    }
    """

    # Name of each of the kernels
    benchmarks = [
        "aes",
        "bulk",
        "crs",
        "kmp",
        "knn",
        "merge",
        "nw",
        "queue",
        "stencil2d",
        "stencil3d",
        "strided"
    ]

    # Create a dict that will contain the processed features of the observation
    features = {}

    # If cpu usage, set the first features to that
    if cpu_usage:
        features["user"] = observation[0][0]
        features["kernel"] = observation[0][1]
        features["idle"] = observation[0][2]

    # Set the first feature, corresponding with the main kernel of the
    # observation
    features["Main"] = observation[-4]

    # Generate a local dict to store observation info
    observation_info = {}

    # Obtain a list with the info of all of the kernels in this observation
    kernels = observation[3].split('_')

    # Iterate over each of the kernels in the observation processing its
    # particular features
    for kernel in kernels:
        # Split the information (name, cu, position of first cu *useless*)
        kernel_keywords = kernel.split('-')
        kernel_name = kernel_keywords[0]
        kernel_accelerators = kernel_keywords[1]
        # Asociate the cus to the kernel name in the temporal dict
        observation_info[kernel_name] = kernel_accelerators

    # Create the features of the observation, each kernel is represented by
    # a feature that indicates the number of cus of that particular kernel in
    # this observation (0 if none)
    for benchmark in benchmarks:
        features[benchmark] = int(observation_info.get(benchmark, 0))

    # Add the top/bottom power and time features of the observation
    features['Power'] = observation[-2]
    features['Time'] = observation[-1]

    return features


def generate_dataframe_from_observations(observations, board, cpu_usage: False):
    """ Creates a dataframe with the observations """

    # Generate list of dictionaries with the features
    tmp_list = []
    for observation in observations:
        if board == "ZCU":
            tmp = process_observation_zcu(observation, cpu_usage)
        elif board == "PYNQ":
            tmp = process_observation_pynq(observation, cpu_usage)
        tmp_list.append(tmp)

    # Create a dataframe from the list of dictionaries
    return pd.DataFrame.from_dict(tmp_list)


def update_dataframe_with_observations(df, observations):
    """ Return the dataframe with the new observations concatenated """

    # Generate a dataframe with the new observations
    df_new = generate_dataframe_from_observations(observations)

    # Concatenate the new dataframe to the previous one and return it
    return pd.concat([df, df_new], ignore_index=True)


def create_update_dataframe_with_observations(observations, df=None):
    """ Return the dataframe with the new observations concatenated
        (creates a new one in case df does not contain one) """

    # Generate a dataframe with the new observations
    df_new = generate_dataframe_from_observations(observations)

    # Return new dataframe if the user do not pass one as argument
    if df is None:
        return df_new

    # Concatenate the new dataframe to the previous one and return it
    return pd.concat([df, df_new], ignore_index=True)
