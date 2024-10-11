#!/usr/bin/env python3

"""
Ping-Pong Buffers Implementation

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : March 2023
Description : This file contains the implementation of the ping-pong buffers
              used for receiving power, traces and online data from an
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
When the python process exits the resource_tracker (which by default registers
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


class PingPongBuffers():
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
                 create=True,
                 unregister=True):
        """
        PingPongBuffers Constructor.\n
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
        tmp_name = "{}_ping_file".format(online_name)
        self._online_ping = shared_memory.SharedMemory(name=tmp_name,
                                                       create=create,
                                                       size=online_size)
        tmp_name = "{}_pong_file".format(online_name)
        self._online_pong = shared_memory.SharedMemory(name=tmp_name,
                                                       create=create,
                                                       size=online_size)
        self._online_current = self._online_ping

        # Power
        tmp_name = "{}_ping_file".format(power_name)
        self._power_ping = shared_memory.SharedMemory(name=tmp_name,
                                                      create=create,
                                                      size=power_size)
        tmp_name = "{}_pong_file".format(power_name)
        self._power_pong = shared_memory.SharedMemory(name=tmp_name,
                                                      create=create,
                                                      size=power_size)
        self._power_current = self._power_ping

        # Traces
        tmp_name = "{}_ping_file".format(traces_name)
        self._traces_ping = shared_memory.SharedMemory(name=tmp_name,
                                                       create=create,
                                                       size=traces_size)
        tmp_name = "{}_pong_file".format(traces_name)
        self._traces_pong = shared_memory.SharedMemory(name=tmp_name,
                                                       create=create,
                                                       size=traces_size)
        self._traces_current = self._traces_ping

    def toggle(self):
        """Toggle the current buffers from ping to pong and vice-versa."""

        if self._online_current is self._online_ping:
            self._online_current = self._online_pong
        else:
            self._online_current = self._online_ping

        if self._power_current is self._power_ping:
            self._power_current = self._power_pong
        else:
            self._power_current = self._power_ping

        if self._traces_current is self._traces_ping:
            self._traces_current = self._traces_pong
        else:
            self._traces_current = self._traces_ping

    def clean(self, unlink=False):
        """Close the shared memories and unlink (remove the files that back
           them) if desired.
        """

        # Close shared memories
        self._online_ping.close()
        self._online_pong.close()
        self._power_ping.close()
        self._power_pong.close()
        self._traces_ping.close()
        self._traces_pong.close()

        # Unlink shared memories (removes from fs)
        if unlink:
            self._online_ping.unlink()
            self._online_pong.unlink()
            self._power_ping.unlink()
            self._power_pong.unlink()
            self._traces_ping.unlink()
            self._traces_pong.unlink()

    @property
    def online(self):
        """Return the current online SharedMemory buffer."""
        return self._online_current.buf

    @property
    def power(self):
        """Return the current power SharedMemory buffer."""
        return self._power_current.buf

    @property
    def traces(self):
        """Return the current traces SharedMemory buffer."""
        return self._traces_current.buf


if __name__ == "__main__":

    # Instantiate each model
    buffers = PingPongBuffers("test_online", 2*1024,
                              "test_power", 525*1024,
                              "test_traces", 20*1024,
                              create=True,
                              unregister=True)

    # Print current files
    print("Online current:", buffers._online_current.name)
    print("Power current:", buffers._power_current.name)
    print("Traces current:", buffers._traces_current.name)

    # Toggle ping to pong
    buffers.toggle()

    # Print current files
    print("Online current:", buffers._online_current.name)
    print("Power current:", buffers._power_current.name)
    print("Traces current:", buffers._traces_current.name)

    # Toggle pong to ping
    buffers.toggle()

    # Print current files
    print("Online current:", buffers._online_current.name)
    print("Power current:", buffers._power_current.name)
    print("Traces current:", buffers._traces_current.name)

    input("Check in '/dev/shm/'")

    buffers.clean(unlink=True)
