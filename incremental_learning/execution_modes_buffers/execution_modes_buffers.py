#!/usr/bin/env python3

"""
Executin Modes Buffers Implementation

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : July 2023
Description : This file contains the implementation of the execution modes
              buffers used for receiving power, traces and online data from an
              independent process over a ram-backed memory-mapped file.
Note        : Requires Python 3.8.x or above.

"""

# Exception if Python version lower than Python 3.8.0
import sys
if sys.version_info < (3, 8, 0):
    raise Exception(
        "Must be using Python 3.8.x or above due to SharedMemories"
    )

# Shared Memory
from multiprocessing import shared_memory
from multiprocessing import resource_tracker
"""
When the python process exits, the resource_tracker (which by default registers
each shared memory block) unlinks by itself the block, so commenting the
unlink() doesn't prevent python for removing the file.
This is a problem because any independent process that uses that shmem block
lose its access, since the file is removed.
(https://bugs.python.org/issue38119)
(https://github.com/python/cpython/pull/15989)
"""


def remove_shm_from_resource_tracker():
    """Monkey-patch multiprocessing.resource_tracker so SharedMemory
    won't be tracked

    More details at: https://bugs.python.org/issue38119
    """

    def fix_register(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.register(self, name, rtype)
    resource_tracker.register = fix_register

    def fix_unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.unregister(self, name, rtype)
    resource_tracker.unregister = fix_unregister

    if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
        del resource_tracker._CLEANUP_FUNCS["shared_memory"]


class ExecutionModesBuffers():
    """Class that creates, manage and cleans the ping-pong buffers used for
    sharing data among independet processes.
    """
    def __init__(self,
                 online_name,
                 online_size,
                 power_name,
                 power_size,
                 traces_name,
                 traces_size,
                 total_iterations,
                 create=True,
                 unregister=True):
        """
        ExecutionModesBuffers Constructor.\n
        Creates two SharedMemories (ping and pong) for power, traces and online
        data that is received from the setup program (c).\n

        - User has to indicate if the files that back the buffers exist
        (create=False) or not (create=True)\n
        - User can choose if the python resource tracker unregisters the
        SharedMemories (unregister=True) or not (unregister=False).\n

        Note: NOT unregistering the SharedMemories means that python will
        remove the files that back the SharedMemories even when not specified
        by the user, making them unaccessible for other processes. (More
        details on: https://bugs.python.org/issue38119)
        """

        # Monkey-patch removing resource tracker SharedMemories registration
        if unregister:
            remove_shm_from_resource_tracker()

        # Create a new shared memory mapped regions (in tmpfs)
        # Online
        tmp_name = "{}_file".format(online_name)
        self._online_base = shared_memory.SharedMemory(
            name=tmp_name,
            create=create,
            size=online_size*total_iterations
        )

        self._online_current_buf = self._online_base.buf[:online_size]
        self._online_size = online_size

        # Power
        tmp_name = "{}_file".format(power_name)
        self._power_base = shared_memory.SharedMemory(
            name=tmp_name,
            create=create,
            size=power_size*total_iterations
        )

        self._power_current_buf = self._power_base.buf[:power_size]
        self._power_size = power_size

        # Traces
        tmp_name = "{}_file".format(traces_name)
        self._traces_base = shared_memory.SharedMemory(
            name=tmp_name,
            create=create,
            size=traces_size*total_iterations
        )

        self._traces_current_buf = self._traces_base.buf[:traces_size]
        self._traces_size = traces_size

        # Store the amount of iterations during the training stage
        self._total_iterations = total_iterations

        # Generate a local variable storing the current training iteration
        self._current_iteration = 0

    def toggle(self):
        """Increment the current iteration counter. Its later used to return
           the memoryview object of the buffer from corresponding address."""

        # Calculate current iteration
        self._current_iteration = \
            (self._current_iteration + 1) % self._total_iterations

        # Online
        # Calculate starting and end of the buffer for the current iteration
        online_starting_point = self._online_size * self._current_iteration
        online_end_point = online_starting_point + self._online_size
        # Slice the buffer
        self._online_current_buf = \
            self._online_base.buf[online_starting_point:online_end_point]

        # Power
        # Calculate starting and end of the buffer for the current iteration
        power_starting_point = self._power_size * self._current_iteration
        power_end_point = power_starting_point + self._power_size
        # Slice the buffer
        self._power_current_buf = \
            self._power_base.buf[power_starting_point:power_end_point]

        # Traces
        # Calculate starting and end of the buffer for the current iteration
        traces_starting_point = self._traces_size * self._current_iteration
        traces_end_point = traces_starting_point + self._traces_size
        # Slice the buffer
        self._traces_current_buf = \
            self._traces_base.buf[traces_starting_point:traces_end_point]

    def clean(self, unlink=False):
        """Close the shared memories and unlink (remove the files that back
           them) if desired.
        """

        # Remove "pointer" to de buffer
        # otherwise python complains when closing the shared memory object
        self._online_current_buf = None
        self._power_current_buf = None
        self._traces_current_buf = None

        # Close shared memories
        self._online_base.close()
        self._power_base.close()
        self._traces_base.close()

        # Unlink shared memories (removes from fs)
        if unlink:
            self._online_base.unlink()
            self._power_base.unlink()
            self._traces_base.unlink()

    @property
    def online(self):
        """Return the current online SharedMemory buffer."""
        return self._online_current_buf

    @property
    def power(self):
        """Return the current power SharedMemory buffer."""
        return self._power_current_buf

    @property
    def traces(self):
        """Return the current traces SharedMemory buffer."""
        return self._traces_current_buf


if __name__ == "__main__":

    test_online_size = 2 * 1024
    test_power_size = 525 * 1024
    test_traces_size = 20 * 1024
    test_total_iterations = 15

    # Instantiate each model
    buffers = ExecutionModesBuffers("test_online", test_online_size,
                                    "test_power", test_power_size,
                                    "test_traces", test_traces_size,
                                    total_iterations=test_total_iterations,
                                    create=True,
                                    unregister=True)

    # Initialize data
    for i in range(0, test_online_size*test_total_iterations, 4):
        value = i // test_online_size
        buffers._online_base.buf[i:i+4] = value.to_bytes(4, sys.byteorder)

    for i in range(0, test_power_size*test_total_iterations, 4):
        value = i // test_power_size
        buffers._power_base.buf[i:i+4] = value.to_bytes(4, sys.byteorder)

    for i in range(0, test_traces_size*test_total_iterations, 4):
        value = i // test_traces_size
        buffers._traces_base.buf[i:i+4] = value.to_bytes(4, sys.byteorder)

    for i in range(test_total_iterations):

        # Print current files
        print("Online current:",
              int.from_bytes(buffers.online[:4], sys.byteorder))
        print("Power current:",
              int.from_bytes(buffers.power[:4], sys.byteorder))
        print("Traces current:",
              int.from_bytes(buffers.traces[:4], sys.byteorder))

        # Toggle to next buffer
        buffers.toggle()

    # Print current files
    print("Online current:",
          int.from_bytes(buffers.online[:4], sys.byteorder))
    print("Power current:",
          int.from_bytes(buffers.power[:4], sys.byteorder))
    print("Traces current:",
          int.from_bytes(buffers.traces[:4], sys.byteorder))

    input("Check in '/dev/shm/'")

    buffers.clean(unlink=True)
