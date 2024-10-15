"""
Extracts observations from an online_info file

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : October 2022
Description : Functions on this file read online_info.bin and traces files to
              generate the corresponding observations for that particular
              time window.
"""


import itertools

from .process_monitor import (
    extract_monitoring_window_info,
    generate_bar_diagram_data_process,
    monitor_data_to_ms,
)
from .process_kernel_combinations import (
    get_kernel_combinations_in_time_window,
    find_executed_kernel,
)
from .process_traces import (
    fragmentate_monitor_measurements,
)
from .generate_observations import (
    get_average_observation_features_from_data,
)


def generate_observations_from_measurement_window(monitor_window,
                                                     slot_list,
                                                     power_buffer,
                                                     traces_buffer,
                                                     board,
                                                     cpu_usage_data):
    """
    (OPTIMIZED with Numpy) Generate CON.BIN and SIG.BIN files for each kernel
    combination in a given monitor measurement (main CON.BIN and SIG.BIN)

    np.convolve(x, np.ones(w), 'valid') / w
    path to the input files:
        CON path -> input_files_path + "/CON" + input_files_id + ".BIN"
        SIG path -> input_files_path + "/SIG" + input_files_id + ".BIN"
        (inside fragmentate_monito_measurements())
    """

    # Get the kernel combinations in the monitor measurement
    relative_kernel_combinations = \
        get_kernel_combinations_in_time_window(slot_list, monitor_window)

    # Slice the CON.BIN and SIG.BIN files to contain each of the kernel
    # combinations
    [power_values_lists,\
        traces_x_values_lists,\
        traces_y_values_lists] = \
        fragmentate_monitor_measurements(
            power_buffer,
            traces_buffer,
            relative_kernel_combinations,
            board
        )

    # Check if the data is relevant enough to be packed
    # If there are less that 4 executions of the kernel, don't care about it
    #
    # relative_kernel_combinations format:
    # [
    #  start_time, end_time, [kernel1,kernel2,...,kerneln],
    #  [[positions_accs_kernel1],[positions_accs_kernel2],...,
    #  [positions_accs_kerneln]]
    # ]
    #

    relevant_executions = []
    for i, combination in enumerate(traces_y_values_lists):
        relevant_executions.append([])
        for j, slot in enumerate(combination[::2]):
            empty = True
            relevant = True
            num_events = len(slot)
            #print(slot)
            #print(len(slot))
            #print(j)
            #print(combination)
            if num_events > 1:
                empty = False
            if num_events > 1 and num_events < 7:#41:#21:#7:
                relevant = False
            if not empty and relevant:
                relevant_executions[i].append(j)

    # Check if the combinations are relevant based on the relevant executions
    # and the suposed kernel combinations and create the file sufix for the
    # model features generation
    relevant_combinations = []
    file_name_sufixes = []
    kernel_names = [
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
        "strided",
    ]

    for relevant_execution, relative_kernel_combination in zip(
                                                relevant_executions,
                                                relative_kernel_combinations):
        # Check if the combination is relevant
        slots_used = list(itertools.chain.from_iterable(
                                            relative_kernel_combination[3]))

        if relevant_execution == slots_used:
            relevant_combinations.append(True)
        else:
            relevant_combinations.append(False)

        # Generate combination sufix
        if relevant_combinations[-1] is True:
            file_name_sufixes.append("")
            for i, kernel in enumerate(relative_kernel_combination[2]):
                # Local variables
                kernel_name = kernel_names[kernel]
                num_parallel_accs = len(relative_kernel_combination[3][i])
                first_acc_position = relative_kernel_combination[3][i][0]
                # Sufix generation
                # format: kernelName1_NumAccs1_FirstAccPosition1_Same4Kernel2_
                # ..._SameForKernelN
                file_name_sufixes[-1] += "{}-{}-{}_".format(kernel_name,
                                                            num_parallel_accs,
                                                            first_acc_position)

            # Remove last '_'
            file_name_sufixes[-1] = file_name_sufixes[-1][:-1]

    # Generate observations
    relevant_combination_count = 0
    observations = []
    #print(len(relevant_combinations))
    #print(relevant_combinations)
    for i, relevant in enumerate(relevant_combinations):

        if not relevant:
            continue

        observation_name = file_name_sufixes[relevant_combination_count]
        # Split the observation name to obtain info about each of the
        # kernels executed in that observation
        kernels = observation_name.split('_')

        # For each of the kernels generate an specific observation
        # (main_tag, top_power, bottom_power, execution_time)
        for kernel in kernels:
            main_tag = kernel_names.index(kernel.split('-')[0])
            accelerator_position = int(kernel.split('-')[2])

            tmp_power_values_lists = [sublist[i] for sublist in power_values_lists]
            tmp_trace_values_lists = [traces_x_values_lists[i], traces_y_values_lists[i]]
            tmp_obs = get_average_observation_features_from_data(
                        observation_name,
                        tmp_power_values_lists,
                        tmp_trace_values_lists,
                        accelerator_position,
                        board)

            tmp_obs.insert(0, main_tag)  # Add main tag to the beginning

            if cpu_usage_data is not None:
                cpu_usage_data = [round(cpu_usage_data[0], 3), round(cpu_usage_data[1], 3), round(cpu_usage_data[2], 3)]
                tmp_obs.insert(0, cpu_usage_data)
            
            #print("cpu_usage_data enable: {}".format(cpu_usage_data is not None))
            #print(tmp_obs)

            observations.append(tmp_obs)

        relevant_combination_count += 1

    # Return the number of kernel combinations, i.e. the number of slices
    # generated

    # print(relative_kernel_combinations)
    return len(relative_kernel_combinations), \
        relevant_combinations.count(False), \
        observations


def generate_observations_from_online_data(online_data_buffer,
                                           power_buffer,
                                           traces_buffer,
                                           board,
                                           cpu_usage: False):
    """ Returns a list of observations (in dictionary format) extracted from
        online info stored in a memory-mapped region
    """

    # Extract monitor window data from file
    cpu_usage_data, \
        monitor_data, \
        slot_list = \
        extract_monitoring_window_info(online_data_buffer, cpu_usage)

    # Create de slots bar diagram data
    slots_data = generate_bar_diagram_data_process(slot_list, monitor_data)

    # Format measurement window info
    measurement_window = monitor_data_to_ms(monitor_data)

    # Find the kernels executed in the measurement window
    executed_kernels = find_executed_kernel(slots_data, measurement_window)

    # Extract observations from the measurement window
    #print("estamos aqui")
    _, _, generated_obs = generate_observations_from_measurement_window(
                            measurement_window,
                            executed_kernels,
                            power_buffer,
                            traces_buffer,
                            board,
                            cpu_usage_data)

    return generated_obs
