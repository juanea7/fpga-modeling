"""
Process monitor information from a monitor_info.bin file generate by the setup

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : October 2022
Description : Functions on this file read a monitor_info.bin file an extract
              and process the monitor information stored inside.
"""

from collections import namedtuple
from multimethod import multimethod
import struct


# Declaration of data structures

name = "TimeSpec"
attrs = [
    "sec",
    "nsec"
]
TimeSpec = namedtuple(name, attrs)  # Stores time by seconds and nanoseconds

name = "MonitorData"
attrs = [
    "initial_time",
    "measured_start_time",
    "measured_finish_time"
]
MonitorData = namedtuple(name, attrs)  # Stores info about monitoring window

name = "KernelData"
attrs = [
    "kernel_id",
    "arrival_time",
    "finish_time"
]
KernelData = namedtuple(name, attrs)  # Stores info about a kernel execution


# Functions


def time_diff(start, end):
    """Calculate the time elapsed between start and end in cycles."""

    sec = 0
    nsec = 0

    if ((end.nsec-start.nsec) < 0):

        sec = end.sec-start.sec-1
        nsec = 1000000000+end.nsec-start.nsec

    else:

        sec = end.sec-start.sec
        nsec = end.nsec-start.nsec

    return TimeSpec(sec, nsec)


def time_to_ms(time):
    """Convert time from Timespec namedtuple to ms."""
    return time.sec*1000 + time.nsec/1000000


def monitor_data_to_ms(monitor_data):
    """Convert monitor data to ms"""
    return (0., time_to_ms(time_diff(monitor_data.measured_start_time,
                                     monitor_data.measured_finish_time)))


def generate_bar_diagram_data_process(slots, monitor_data):
    """Generate gantt diagram data from slots to process obs."""
    diagram_data_list = []

    for i, slot in enumerate(slots):

        # Create a list per slot
        diagram_data_list.append([])

        for kernel in slot:

            # Compute meaningful data for each kernel in the slot
            arrival_time = time_diff(monitor_data.measured_start_time,
                                     kernel.arrival_time)
            finish_time = time_diff(monitor_data.measured_start_time,
                                    kernel.finish_time)
            arrival_time_ms = time_to_ms(arrival_time)
            finish_time_ms = time_to_ms(finish_time)

            # Append computed kernel data to the slot list
            # ([(start, finish)],kernel_id)
            diagram_data_list[i].append(
                (kernel.kernel_id, arrival_time_ms, finish_time_ms)
            )

    return diagram_data_list


@multimethod
def extract_monitoring_window_info(filename: str, cpu_usage: bool, board: dict):
    """Extracts info of the monitoring window from a file.
       Info such as window start and stop time, kernels per slot, etc.

       Structure of the file


        user_cpu | kernel_cpu | idle_cpu | monitor_window_data | ->

        ->                 | next is a kernel| k | ->
        -> number_of_slots |        1        | . | ->

        -> next is a kernel| k | next is a kernel| k | next is a kernel| k | ->
        ->        1        | . |        1        | . |        1        | . | ->


        -> next slot | next is a kernel| k | next is a kernel| k | next slot |>
        ->     0     |        1        | . |        1        | . |     0     |>

        -> Repeted num_slots times |
        -> ....................... |

       Data structures behind the names of the above diagram:

       typedef struct{
           timespec initial_time; (2 long int)
           timespec start_time; (2 long int)
           timespec end_time; (2 long int)
       }monitor_window_data;

       typedef struct{
           int kernel_id; (1 int)
           timespec arrival_time; (2 long int)
           timespec end_time; (2 long int)
       }k;

       int number_of_slots; (1 int)

       int separator; (1 int)

       Return:

       - cpu_usage    # Tuple with the user, kernel and idle time of the cpu
       - monitor_data # Stores info about the monitoring window
       - slot_list    # Contains a list per slot with every kernel executed
                      # in it chronologically
    """

    # Format of the data structures inside the file
    if board["arch"] == "64bit":
        # Use native size (8 bytes for long (l), same on ZCU timespec)
        cpu_usage_format = "f"
        monitor_data_format = "6l"
        kernel_data_format = "1i4l"
        separator_format = "i"
    elif board["arch"] == "32bit":
        # Use standard size (4 bytes for long (l), same on PYNQ timespec)
        cpu_usage_format = "=f"
        monitor_data_format = "=6l"
        kernel_data_format = "=1i4l"
        separator_format = "=i"
    else:
        raise ValueError(f"Board['arch'] {board['arch']} not supported.")

    # Calculate the size of each data structure
    cpu_usage_size = struct.calcsize(cpu_usage_format)
    monitor_data_size = struct.calcsize(monitor_data_format)
    kernel_data_size = struct.calcsize(kernel_data_format)
    separator_size = struct.calcsize(separator_format)

    # Variables declaration
    cpu_usage_data = []
    monitor_data = MonitorData  # Stores data of the monitoring window
    kernel_data = KernelData    # Stores data of a kernel execution
    separator_data = 0          # Stores the separator data
    slot_list = []              # Contains all the slots on monitoring window
    num_slots = 0               # Stores the number of slots

    # Open monitor info binary file in binary mode
    with open(filename, "rb") as file:

        if cpu_usage:
            # Pasar una estructura en lugar de 3 float?
            # Read user cpu_usage
            data = file.read(cpu_usage_size)
            # Exits if there is no data read
            if not data:
                print("Error user cpu usage read\n")
                exit(1)
            # Unpack the data in structure-like format and assign to variable
            cpu_usage_data.append(struct.unpack(cpu_usage_format, data)[0])

            # Read kernel cpu_usage
            data = file.read(cpu_usage_size)
            # Exits if there is no data read
            if not data:
                print("Error user cpu usage read\n")
                exit(1)
            # Unpack the data in structure-like format and assign to variable
            cpu_usage_data.append(struct.unpack(cpu_usage_format, data)[0])

            # Read idle cpu_usage
            data = file.read(cpu_usage_size)
            # Exits if there is no data read
            if not data:
                print("Error user cpu usage read\n")
                exit(1)
            # Unpack the data in structure-like format and assign to variable
            cpu_usage_data.append(struct.unpack(cpu_usage_format, data)[0])

        else:
            cpu_usage_data = None

        # Read monitoring window data from file
        data = file.read(monitor_data_size)

        # Exits if there is no data read
        if not data:
            print("Error monitor data read\n")
            exit(1)

        # Unpack the data in structure-like format
        monitor_data_raw = struct.unpack(monitor_data_format, data)

        # assign data to the namedtuple
        monitor_data = MonitorData(TimeSpec(monitor_data_raw[0],
                                            monitor_data_raw[1]),
                                   TimeSpec(monitor_data_raw[2],
                                            monitor_data_raw[3]),
                                   TimeSpec(monitor_data_raw[4],
                                            monitor_data_raw[5]))

        # Read number of ARTICo3 slots used
        data = file.read(separator_size)

        # Exits if there is no data read
        if not data:
            print("Error artico3 slots read\n")
            exit(1)

        # Unpack the data in structure-like format and assign to variable
        num_slots = struct.unpack(separator_format, data)[0]

        # Process each kernel in each slot a create the following list:
        #
        # slot_list = [slot_0_list, ... , slot_n_list]
        #
        #       slot_0_list = [kernel_data_0, ... , kernel_data_n]
        #
        for i in range(num_slots):

            kernel_data_list = []

            # For each slot get every kernel data
            while True:

                # Get separator data from file
                data = file.read(separator_size)

                # Exits if there is no data read
                if not data:
                    print("Error file separator read\n")
                    exit(1)

                # Unpack the separator data in structure-like format and
                # assign to a variable
                separator_data = struct.unpack(separator_format, data)[0]

                # If the separator contains a 0 means there are no more
                # kernels in this slots (so jump to next slot)
                # If the separator contains a 1 there are more kernels yet
                if separator_data == 0:
                    break

                # Get kernel data from file
                data = file.read(kernel_data_size)

                # Exits if there is no data read
                if not data:
                    print("Error kernel data read\n")
                    exit(1)

                # Unpack the kernel data in a structure-like format
                kernel_data = struct.unpack(kernel_data_format, data)

                # Assign data to the namedtuple and append it to a list
                kernel_data_list.append(KernelData(kernel_data[0],
                                                   TimeSpec(kernel_data[1],
                                                            kernel_data[2]),
                                                   TimeSpec(kernel_data[3],
                                                            kernel_data[4])))

            # Add the list of kernels to the corresponding slot
            slot_list.append(kernel_data_list)

    return cpu_usage_data, monitor_data, slot_list


@multimethod
def extract_monitoring_window_info(buffer: memoryview, cpu_usage: bool, board: dict):
    """Extracts info of the monitoring window from a file.
       Info such as window start and stop time, kernels per slot, etc.

       Structure of the file


        user_cpu | kernel_cpu | idle_cpu | monitor_window_data | ->

        ->                 | next is a kernel| k | ->
        -> number_of_slots |        1        | . | ->

        -> next is a kernel| k | next is a kernel| k | next is a kernel| k | ->
        ->        1        | . |        1        | . |        1        | . | ->


        -> next slot | next is a kernel| k | next is a kernel| k | next slot |>
        ->     0     |        1        | . |        1        | . |     0     |>

        -> Repeted num_slots times |
        -> ....................... |

       Data structures behind the names of the above diagram:

       typedef struct{
           timespec initial_time; (2 long int)
           timespec start_time; (2 long int)
           timespec end_time; (2 long int)
       }monitor_window_data;

       typedef struct{
           int kernel_id; (1 int)
           timespec arrival_time; (2 long int)
           timespec end_time; (2 long int)
       }k;

       int number_of_slots; (1 int)

       int separator; (1 int)

       Return:

       - monitor_data # Stores info about the monitoring window
       - slot_list    # Contains a list per slot with every kernel executed
                      # in it chronologically
    """

    # Format of the data structures inside the file
    if board["arch"] == "64bit":
        # Use native size (8 bytes for long (l), same on ZCU timespec)
        cpu_usage_format = "f"
        monitor_data_format = "6l"
        kernel_data_format = "1i4l"
        separator_format = "i"
    elif board["arch"] == "32bit":
        # Use standard size (4 bytes for long (l), same on PYNQ timespec)
        cpu_usage_format = "=f"
        monitor_data_format = "=6l"
        kernel_data_format = "=1i4l"
        separator_format = "=i"
    else:
        raise ValueError(f"Board['arch'] {board['arch']} not supported.")

    # Calculate the size of each data structure
    cpu_usage_size = struct.calcsize(cpu_usage_format)
    monitor_data_size = struct.calcsize(monitor_data_format)
    kernel_data_size = struct.calcsize(kernel_data_format)
    separator_size = struct.calcsize(separator_format)

    # Variables declaration
    cpu_usage_data = []
    monitor_data = MonitorData  # Stores data of the monitoring window
    kernel_data = KernelData    # Stores data of a kernel execution
    separator_data = 0          # Stores the separator data
    slot_list = []              # Contains all the slots on monitoring window
    num_slots = 0               # Stores the number of slots
    index = 0                   # Stores the buffer index used to travers it

    if cpu_usage:

        # Get user_cpu_usage data from file
        data = buffer[index:index+cpu_usage_size]
        index += cpu_usage_size
        # Unpack the cpu_usage data in structure-like format and
        # assign to a variable
        cpu_usage_data.append(struct.unpack(cpu_usage_format, data)[0])

        # Get kernel_cpu_usage data from file
        data = buffer[index:index+cpu_usage_size]
        index += cpu_usage_size
        # Unpack the cpu_usage data in structure-like format and
        # assign to a variable
        cpu_usage_data.append(struct.unpack(cpu_usage_format, data)[0])

        # Get idle_cpu_usage data from file
        data = buffer[index:index+cpu_usage_size]
        index += cpu_usage_size
        # Unpack the cpu_usage data in structure-like format and
        # assign to a variable
        cpu_usage_data.append(struct.unpack(cpu_usage_format, data)[0])
    else:
        cpu_usage_data = None

    # Read monitoring window data from file
    data = buffer[index:index+monitor_data_size]
    index += monitor_data_size

    # Unpack the data in structure-like format
    monitor_data_raw = struct.unpack(monitor_data_format, data)

    # assign data to the namedtuple
    monitor_data = MonitorData(TimeSpec(monitor_data_raw[0],
                                        monitor_data_raw[1]),
                               TimeSpec(monitor_data_raw[2],
                                        monitor_data_raw[3]),
                               TimeSpec(monitor_data_raw[4],
                                        monitor_data_raw[5]))

    # Read number of ARTICo3 slots used
    data = buffer[index:index+separator_size]
    index += separator_size

    # Unpack the data in structure-like format and assign to variable
    num_slots = struct.unpack(separator_format, data)[0]

    # Process each kernel in each slot a create the following list:
    #
    # slot_list = [slot_0_list, ... , slot_n_list]
    #
    #       slot_0_list = [kernel_data_0, ... , kernel_data_n]
    #
    for i in range(num_slots):

        kernel_data_list = []

        # For each slot get every kernel data
        while True:

            # Get separator data from file
            data = buffer[index:index+separator_size]
            index += separator_size

            # Unpack the separator data in structure-like format and
            # assign to a variable
            separator_data = struct.unpack(separator_format, data)[0]

            # If the separator contains a 0 means there are no more
            # kernels in this slots (so jump to next slot)
            # If the separator contains a 1 there are more kernels yet
            if separator_data == 0:
                break

            # Get kernel data from file
            data = buffer[index:index+kernel_data_size]
            index += kernel_data_size

            # Unpack the kernel data in a structure-like format
            kernel_data = struct.unpack(kernel_data_format, data)

            # Assign data to the namedtuple and append it to a list
            kernel_data_list.append(KernelData(kernel_data[0],
                                    TimeSpec(kernel_data[1],
                                             kernel_data[2]),
                                    TimeSpec(kernel_data[3],
                                             kernel_data[4])))

        # Add the list of kernels to the corresponding slot
        slot_list.append(kernel_data_list)

    return cpu_usage_data, monitor_data, slot_list
