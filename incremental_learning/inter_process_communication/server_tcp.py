#!/usr/bin/env python3
"""
Unix Domain TCP socket - server implementation

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : October 2022
Description : This file contains a class that implements a Unix-domain TCP
              socket for the server side.
"""

import socket
import os


class ServerSocketTCP:
    """Its a TCP server socket"""

    def __init__(self, name, input_size, output_size):
        """create the socket"""

        self.server_address = "/tmp/{}".format(name)
        self.input_size = input_size
        self.output_size = output_size

        # Make sure the socket does not already exist
        try:
            os.unlink(self.server_address)
        except OSError:
            if os.path.exists(self.server_address):
                raise

        # Create a UDP socket
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

        # Bind the socket to the address
        self.socket.bind(self.server_address)

        # Listening for incomming connections
        self.socket.listen(1)

    def wait_connection(self):
        """wait for the client to connect"""
        # Wait for a client to connect
        self.connection, self.client_address = self.socket.accept()

    def recv_data(self):
        """get data from client"""
        # Receive data from client
        return self.connection.recv(self.input_size)

    def send_data(self, data):
        """send data to client"""
        # Send data to the client
        self.connection.sendall(data)

    def close_connection(self):
        "close the server-client connection"
        self.connection.close()

    def __del__(self):
        """close the socket and clean up"""
        # Close the socket and remove the file
        self.socket.close()
        os.unlink(self.server_address)


if __name__ == "__main__":

    # Create UDT server socket
    tcp_socket = ServerSocketTCP(20, 20)

    tcp_socket.wait_connection()
    data = tcp_socket.recv_data()
    tcp_socket.send_data(data)

    tcp_socket.close_connection()
