#!/usr/bin/env python3
"""
Unix Domain UDP socket - server implementation

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : October 2022
Description : This file contains a class that implements a Unix-domain UDP
              socket for the server side.
"""

import socket
import os


class ServerSocketUDP:
    """Its a user datagram protocol server socket"""

    def __init__(self, name, input_size):
        """creates the socket"""

        self.server_address = "/tmp/{}".format(name)
        self.input_size = input_size

        # Make sure the socket does not already exist
        try:
            os.unlink(self.server_address)
        except OSError:
            if os.path.exists(self.server_address):
                raise

        # Create a UDP socket
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)

        # Bind the socket to the address
        self.socket.bind(self.server_address)

    def wait_notification(self):
        """Wait for a notification from the client"""

        # Wait for the notification
        data = self.socket.recvfrom(self.input_size)

        # Return notification
        return data[0]

    def __del__(self):
        """closes the socket and clean up"""

        # Close the socket and remove the file
        self.socket.close()
        os.unlink(self.server_address)


if __name__ == "__main__":

    # Create UDT server socket
    tcp_socket = ServerSocketUDP(1)

    notification = tcp_socket.wait_notification()
    print("This is the notification: {}".format(notification))
