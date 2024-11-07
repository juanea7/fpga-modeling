"""
Compute observations out of power consumption and performance data

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : October 2022
Description : Functions on this file calculate observations (number and type
              of kernels, as well as power consumption and performance) from
              a given kernel combination and power and performance traces.
"""

import numpy as np

from .find_elements import (
    get_closest_element_above,
    get_closest_element_below,
)


def calculate_continuous_series_arithmetic_mean(my_list):
    """ Continuous Series Arithmetic Mean - Direct Method """

    # Calculate the middle point of each group.
    # Groups -> (e[0]-e[1]), (e[1]-e[2])...
    # groups_middle_points = \
    #   np.mean( np.array([ my_list[:-1], my_list[1:] ]), axis=0 )
    # A rolling average of window 2
    groups_middle_points = np.convolve(my_list, np.ones(2), 'valid') / 2

    # Calculate the mean of the whole list of middle points
    # (it should be performed a sumatory of each middle point times its
    # frequency of appearence and divided by the sumatory of que frequencies,
    # but since each group presents the same frequency we can just perform
    # the mean of the list of middle points)
    # URL -> https://short.upm.es/00h0h
    return np.mean(groups_middle_points)


def get_average_observation_features_from_data_power_dual(observation_name,
                                                      top_power_x_values,
                                                      top_power_y_values,
                                                      bottom_power_x_values,
                                                      bottom_power_y_values,
                                                      traces_x_values_list,
                                                      traces_y_values_list,
                                                      accelerator_position):
    """ Obtains model features from observations
        (performing a mean between all the available executions
    """

    # Print interesting information
    # print("-- Name: {} - Samples: {} - Position: {} --".format(input_path,
    #                                                     num_samples,
    #                                                     accelerator_position))

    # traces_x_values_list format: [[sig0],[sig1],...,[sigN]]  (tiempos)
    #                              [[start0],[ready0],[start1],[stop1],...[startN],[stopN]]
    # traces_y_values_list format: [[sig0],[sig1],...,[sigN]]  (valores)
    #                              [[start0],[ready0],[start1],[stop1],...[startN],[stopN]]

    # Get the data of the accelerator to get the features from
    selected_traces_x_values = traces_x_values_list[2 * accelerator_position]
    selected_traces_y_values = traces_y_values_list[2 * accelerator_position]
    # 2 * accelerator_position because the even elements are the
    # start signals, and the execution happens between starts
    # the odd elements are the ready signals, useless

    # Calculate mean features

    # Calculate number of executions and start of the first and end of the
    # last one
    #

    # Get the first non-zero value index
    start_index = np.min(np.nonzero(selected_traces_y_values)[0])
    # Get the last non-zero value index
    end_index = np.max(np.nonzero(selected_traces_y_values)[0])
    num_executions = (end_index - start_index) / 2

    # Find the closest elements of the power consumption traces
    start_power_index = \
        get_closest_element_above(top_power_x_values,
                                  selected_traces_x_values[start_index])
    end_power_index = \
        get_closest_element_below(top_power_x_values,
                                  selected_traces_x_values[end_index])

    # Calculate features
    # (continuous_series_arithmetic_mean has been used to avoid doing
    # interpolation reducing drastically the execution time)
    tmp = top_power_y_values[start_power_index:end_power_index]
    average_top_power = calculate_continuous_series_arithmetic_mean(tmp)
    tmp = bottom_power_y_values[start_power_index:end_power_index]
    average_bottom_power = calculate_continuous_series_arithmetic_mean(tmp)
    execution_time = (selected_traces_x_values[end_index] -
                      selected_traces_x_values[start_index]) / num_executions

    # Round to 3 decimals
    average_top_power = round(average_top_power, 3)
    average_bottom_power = round(average_bottom_power, 3)
    execution_time = round(execution_time, 3)

    # print("Average TOP Power Consumption (W):", average_top_power)
    # print("Average BOTTOM Power Consumption (W):", average_bottom_power)
    # print("Execution Time (ms):", execution_time)

    # print("Observation Name:", observation_name)

    return [observation_name,
            average_top_power,
            average_bottom_power,
            execution_time]


def get_average_observation_features_from_data_power_mono(observation_name,
                                                       power_x_values,
                                                       power_y_values,
                                                       traces_x_values_list,
                                                       traces_y_values_list,
                                                       accelerator_position):
    """ Obtains model features from observations
        (performing a mean between all the available executions
    """

    # Print interesting information
    # print("-- Name: {} - Samples: {} - Position: {} --".format(input_path,
    #                                                     num_samples,
    #                                                     accelerator_position))

    # traces_x_values_list format: [[sig0],[sig1],...,[sigN]]  (tiempos)
    #                              [[start0],[ready0],[start1],[stop1],...[startN],[stopN]]
    # traces_y_values_list format: [[sig0],[sig1],...,[sigN]]  (valores)
    #                              [[start0],[ready0],[start1],[stop1],...[startN],[stopN]]

    # Get the data of the accelerator to get the features from
    selected_traces_x_values = traces_x_values_list[2 * accelerator_position]
    selected_traces_y_values = traces_y_values_list[2 * accelerator_position]
    # 2 * accelerator_position because the even elements are the
    # start signals, and the execution happens between starts
    # the odd elements are the ready signals, useless

    # Calculate mean features

    # Calculate number of executions and start of the first and end of the
    # last one
    #

    # Get the first non-zero value index
    start_index = np.min(np.nonzero(selected_traces_y_values)[0])
    # Get the last non-zero value index
    end_index = np.max(np.nonzero(selected_traces_y_values)[0])
    num_executions = (end_index - start_index) / 2

    # Find the closest elements of the power consumption traces
    start_power_index = \
        get_closest_element_above(power_x_values,
                                  selected_traces_x_values[start_index])
    end_power_index = \
        get_closest_element_below(power_x_values,
                                  selected_traces_x_values[end_index])

    # Calculate features
    # (continuous_series_arithmetic_mean has been used to avoid doing
    # interpolation reducing drastically the execution time)
    tmp = power_y_values[start_power_index:end_power_index]
    average_power = calculate_continuous_series_arithmetic_mean(tmp)
    execution_time = (selected_traces_x_values[end_index] -
                      selected_traces_x_values[start_index]) / num_executions

    # Round to 3 decimals
    average_power = round(average_power, 3)
    execution_time = round(execution_time, 3)

    # print("Average Power Consumption (W):", average_power)
    # print("Execution Time (ms):", execution_time)

    # print("Observation Name:", observation_name)

    return [observation_name,
            average_power,
            execution_time]


# Map board to functions
observation_functions = {
    "dual": get_average_observation_features_from_data_power_dual,
    "mono": get_average_observation_features_from_data_power_mono
}

def get_average_observation_features_from_data(observation_name,
                                                power_values_lists,
                                                traces_values_lists,
                                                accelerator_position,
                                                board):
    """ Obtains model features from observations
        (performing a mean between all the available executions
    """

    if board["power"]["rails"] not in observation_functions:
        raise ValueError(f"Board['power']['rails'] not recognized: {board['power']['rails']}")

    return observation_functions[board["power"]["rails"]](observation_name,
                                                          *power_values_lists,
                                                          *traces_values_lists,
                                                          accelerator_position)
