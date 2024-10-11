"""
Obtain the combinations of kernels

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : October 2022
Description : Functions on this file identify which are the combinations of
              kernels that happen in a given time window.
"""

from copy import deepcopy

from .find_elements import get_closest_element_below_item


def find_executed_kernel(slot_list, measurement_window):
    """
    (OPTIMIZED)
    Return a list indicating for a given interval of time, which kernel is
    executed on each slot and its visibility window (in which fragment of
    that given interval of time it was executing
    """

    output = [[] for i in range(len(slot_list))]

    measurement_start_time = measurement_window[0]
    measurement_finish_time = measurement_window[1]

    for i, slot in enumerate(slot_list):
        # Instead of iterate over all the executions in the list
        # we will create a slice from the kernel that starts before the start
        # time and end after the end time
        # slot list has ordered executions ***
        first_start_index = \
            get_closest_element_below_item(slot,
                                           measurement_start_time,
                                           1)
        last_start_index = \
            get_closest_element_below_item(slot,
                                           measurement_finish_time,
                                           1)
        executions_to_test = slot[first_start_index:last_start_index+1]

        for execution in executions_to_test:

            kernel_id = execution[0]
            execution_start = execution[1]
            execution_finish = execution[2]

            if execution_start < measurement_finish_time and \
               execution_finish > measurement_start_time:
                output[i].append(
                    (kernel_id,
                     max(execution_start, measurement_start_time),
                     min(execution_finish, measurement_finish_time))
                )

    return output


def get_kernel_combination_info(kernel_combination):
    """ Returns info about a given kernel combination with this format:

        (start_time, stop_time, [kernel1,kernel2,...,kerneln],
        [[positions kernel1],[positions kernel2],...,[positions kerneln]])

    """

    for kernel in kernel_combination:
        if len(kernel) > 0:
            start_time = kernel[0][1]
            end_time = kernel[0][2]
            break

    kernels_info = {}

    for i, kernel in enumerate(kernel_combination):
        if len(kernel) > 0:
            if kernel[0][0] in kernels_info:
                kernels_info[kernel[0][0]].append(i)
            else:
                kernels_info[kernel[0][0]] = [i]

    return (start_time,
            end_time,
            list(kernels_info),
            list(kernels_info.values()))


def get_kernel_combinations_in_time_window(slot_list, time_window):
    """ Get the kernel combinations inside a specific time window in
        this format:

        kernel_combinations data structure = [
            (start_time,
             stop_time,
             [kernel1,kernel2,...,kerneln],
             [[positions kernel1],
             [positions kernel2],
             ..., [positions kerneln]]),
             ... tantos elementos como combinaciones diferentes de kernels haya
        ]
    """
    # If you do a normal asignment you do a shallow copy (not a copy but a
    # pointer asignment), you need to deep copy
    aux = deepcopy(slot_list)
    start_offset = 0
    """
    Stages data structur = [
        (start_time, stop_time,
        [kernel1,kernel2,...,kerneln],
        [[positions kernel1],
        [positions kernel2],
        ...,[positions kerneln]]),
        ...(tantos elemento como etapas diferentes haya)
    ]
    """

    kernel_combinations = []

    while (True):

        tmp = time_window[1]

        # Slot_list = [slot_0, ... , slot_n]
        # slot_0 = [kernel_0, ... , kernel_n]
        # kernel_0 = (kernel_id, arrival, finish)

        # Find the first start event
        index_slot = None
        index_kernel = None
        for i, slot in enumerate(aux):  # For each slot
            for j, kernel in enumerate(slot):  # For each kernel
                if len(slot) > 0 and \
                   kernel[1] < tmp and \
                   kernel[2] > start_offset:
                    tmp = kernel[1]
                    index_slot = i
                    index_kernel = j
                    continue

        if index_slot is None:
            break
        start_time = aux[index_slot][index_kernel][1]
        if start_time < start_offset:
            start_time = start_offset

        # Get first stop among the kernel that start at this same time
        tmp = aux[index_slot][index_kernel][2]

        index_slot = None
        index_kernel = None
        for i, slot in enumerate(aux):  # For each slot
            for j, kernel in enumerate(slot):  # For each kernel
                if len(slot) > 0 and \
                   kernel[1] <= start_time and \
                   kernel[2] < tmp and \
                   kernel[2] > start_offset:
                    tmp = kernel[2]
                    index_slot = i
                    index_kernel = j
                    continue

        if index_slot is not None:
            tmp = aux[index_slot][index_kernel][2]

        # Check if between the start and stop there has been another start
        # (so there will be a region before and after that start)
        index_slot = None
        index_kernel = None
        for i, slot in enumerate(aux):  # For each slot
            for j, kernel in enumerate(slot):  # For each kernel
                if len(slot) > 0 and \
                   kernel[1] < tmp and \
                   kernel[1] != start_time and \
                   kernel[1] > start_offset:
                    tmp = kernel[1]
                    index_slot = i
                    index_kernel = j
                    continue

        if index_slot is None:
            end_time = tmp
        else:
            end_time = aux[index_slot][index_kernel][1]

        # Find executed kernels in this time window
        kernel_combination = \
            find_executed_kernel(slot_list, (start_time, end_time))
        # kernel_combination = \
        #    find_executed_kernel(slot_list, (start_time, end_time-start_time))

        # Get the info from that kernel_combination
        kernel_combination_info = \
            get_kernel_combination_info(kernel_combination)

        # Append the kernel combination to the list
        kernel_combinations.append(kernel_combination_info)

        # Next iteration processes from end_time
        start_offset = end_time

    return kernel_combinations
