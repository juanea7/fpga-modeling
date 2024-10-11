#!/usr/bin/env python3
"""
Inet Domain TCP socket - server implementation

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : July 2023
Description : This file contains a class that implements a Inet-domain TCP
              socket for the server side.
"""

from encodings import utf_8
import socket
import sys
import ctypes as ct
import time
import argparse


class BufferInfo(ct.Structure):
    """ Buffer Info for socket transmission - This class defines a C-like struct """
    _fields_ = [
        ("num_packets", ct.c_int),
        ("regular_packet_size", ct.c_int),
        ("last_packet_size", ct.c_int)
    ]

    def get_dict(self):
        """Convert to dictionary"""
        return dict((f, getattr(self, f)) for f, _ in self._fields_)


class ServerSocketTCP:
    """Its a TCP inet socket"""

    def __init__(self, ip, port):
        """creates the socket"""

        self.server_address = (ip, port)

        # Create a TCP socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Bind the socket to the address
        self.socket.bind(self.server_address)

        # Listening for incomming connections
        self.socket.listen(1)

    def wait_connection(self):
        """wait for the client to connect"""
        # Wait for a client to connect
        self.connection, self.client_address = self.socket.accept()

    def recv_data(self, input_size):
        """get data from client.
           This function ensures input_size bytes are read, since socket.recv()
           does not guarantee it reads the number of bytes you demand."""

        # Internal variables used to ensure all input_size is read
        num_bytes_recv = 0
        data_out = bytes()

        # Read until input_size bytes are received
        while (num_bytes_recv < input_size):

            # Read from socket the remaining number of bytes to be read
            data_aux = self.connection.recv(input_size - num_bytes_recv)

            # Check if the connection is broken
            if data_aux == b'':
                raise RuntimeError("socket connection broken")

            # Append read data to previously read data
            data_out += data_aux

            # Calculate how many bytes are left to be read
            num_bytes_recv = len(data_out)

        return data_out

    def recv_data_old(self, input_size):
        """get data from client"""
        # Receive data from client
        return self.connection.recv(input_size)

    def send_data(self, data):
        """send data to client"""
        # Send data to the client
        self.connection.sendall(data)

    def close_connection(self):
        "close the server-client connection"
        self.connection.close()

    def __del__(self):
        """close the socket and clean up"""
        print("Se está cerrando el socket")
        # Close the socket and remove the file
        self.socket.close()


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()

    # Indicate the path of the file containing the dataset
    parser.add_argument("output_path",
                        help="Path to the folder where traces will be stored")

    args = parser.parse_args(sys.argv[1:])

    # Socket variables
    my_hostname=socket.gethostname()
    socket_ip_address=socket.gethostbyname(my_hostname)
    socket_port = 4242

    print("[Socket Info] {}:{}".format(socket_ip_address, socket_port))

    # Create TCP server socket
    tcp_socket = ServerSocketTCP(socket_ip_address, socket_port)
    print("Socket created")

    # Wait for connection on the client side
    try:
        tcp_socket.wait_connection()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        exit(0)

    print("Client connected!")

    # Aux variables
    i = 0
    bytes_to_recv = 0

    # Infinite operation
    # TODO: handle the interruption of this operation
    while True:

        print("Waiting for incoming buffers...")

        # Get info about the next power buffer
        try:
            data = tcp_socket.recv_data(ct.sizeof(BufferInfo))
        except KeyboardInterrupt:
            print("Keyboard Interrupt")
            exit(0)

        buffer_info_c = BufferInfo.from_buffer_copy(data)
        buffer_info = buffer_info_c.get_dict()

        print("[{}] Power - Number of iterations: {}".format(i, buffer_info["num_packets"]))

        with open("{}/traces/CON_{}.BIN".format(args.output_path, i), "wb") as binary_file:

            # Convert to bytes
            data = bytes()

            # Process each received packet
            for iter in range(buffer_info["num_packets"]):

                # Calculate size to recieve
                num_bytes_to_recv = buffer_info["regular_packet_size"] if iter < buffer_info["num_packets"] - 1 else buffer_info["last_packet_size"]
                #print("iter: {} | bytes_to_recv: {}".format(iter, num_bytes_to_recv))
                data += tcp_socket.recv_data(num_bytes_to_recv)

            # Write bytes to file
            binary_file.write(data)

        # Get info about the next traces buffer
        data = tcp_socket.recv_data(ct.sizeof(BufferInfo))
        buffer_info_c = BufferInfo.from_buffer_copy(data)
        buffer_info = buffer_info_c.get_dict()

        print("[{}] Traces - Number of iterations: {}".format(i, buffer_info["num_packets"]))

        with open("{}/traces/SIG_{}.BIN".format(args.output_path, i), "wb") as binary_file:

            # Conver to bytes
            data = bytes()

            # Process each received packet
            for iter in range(buffer_info["num_packets"]):

                # Calculate size to recieve
                num_bytes_to_recv = buffer_info["regular_packet_size"] if iter < buffer_info["num_packets"] - 1 else buffer_info["last_packet_size"]
                #print("iter: {} | bytes_to_recv: {}".format(iter, num_bytes_to_recv))
                data = tcp_socket.recv_data(num_bytes_to_recv)

            # Write bytes to file
            binary_file.write(data)


        # Get info about the next outputs buffer
        data = tcp_socket.recv_data(ct.sizeof(BufferInfo))
        buffer_info_c = BufferInfo.from_buffer_copy(data)
        buffer_info = buffer_info_c.get_dict()

        print("[{}] Online - Number of iterations: {}".format(i, buffer_info["num_packets"]))

        with open("{}/outputs/online_{}.bin".format(args.output_path, i), "wb") as binary_file:

            # Convert to bytes
            data = bytes()

            for iter in range(buffer_info["num_packets"]):

                # Calculate size to recieve
                num_bytes_to_recv = buffer_info["regular_packet_size"] if iter < buffer_info["num_packets"] - 1 else buffer_info["last_packet_size"]
                #print("iter: {} | bytes_to_recv: {}".format(iter, num_bytes_to_recv))
                data = tcp_socket.recv_data(num_bytes_to_recv)

            # Write bytes to file
            binary_file.write(data)

        # Increment the count
        i += 1
