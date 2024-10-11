#!/usr/bin/env python3

"""
Run-Time Training and Prediction with Online Models

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : October 2022
Description : This script:
              - Processes power and performance traces received from an
                independent c-code process via disk-backed files or ram-backed
                memory-mapped regions (up to the user).
              - Trains power (pl and ps) and performance online models on
                demand from commands send via a socket from the independent
                c-code process.
              - Predicts with the trained models for the features receive from
                the independent c-code process via another socket.
              All is done at run-time with concurrent threads fortraining the
              models and predicting with them.
              TODO: Apply the tcp training socket changes to the ram version
"""

import sys
import os
import argparse
import time
# from multiprocessing import Process
from threading import Thread
from threading import Lock
import ctypes as ct
import pandas as pd
import river

from incremental_learning import online_data_processing as odp
from incremental_learning import inter_process_communication as ipc
from incremental_learning import online_models as om
from incremental_learning import ping_pong_buffers as ppb
from incremental_learning import execution_modes_buffers as emb

from datetime import datetime, timezone

import struct


class FeatureswCPUUsage(ct.Structure):
    """ Features with CPU Usage - This class defines a C-like struct """
    _fields_ = [
        ("user", ct.c_float),
        ("kernel", ct.c_float),
        ("idle", ct.c_float),
        ("aes", ct.c_uint8),
        ("bulk", ct.c_uint8),
        ("crs", ct.c_uint8),
        ("kmp", ct.c_uint8),
        ("knn", ct.c_uint8),
        ("merge", ct.c_uint8),
        ("nw", ct.c_uint8),
        ("queue", ct.c_uint8),
        ("stencil2d", ct.c_uint8),
        ("stencil3d", ct.c_uint8),
        ("strided", ct.c_uint8)
    ]

    def get_dict(self):
        """Convert to dictionary"""
        return dict((f, getattr(self, f)) for f, _ in self._fields_)


class FeatureswoCPUUsage(ct.Structure):
    """ Features without CPU Usage- This class defines a C-like struct """
    _fields_ = [
        ("aes", ct.c_uint8),
        ("bulk", ct.c_uint8),
        ("crs", ct.c_uint8),
        ("kmp", ct.c_uint8),
        ("knn", ct.c_uint8),
        ("merge", ct.c_uint8),
        ("nw", ct.c_uint8),
        ("queue", ct.c_uint8),
        ("stencil2d", ct.c_uint8),
        ("stencil3d", ct.c_uint8),
        ("strided", ct.c_uint8)
    ]

    def get_dict(self):
        """Convert to dictionary"""
        return dict((f, getattr(self, f)) for f, _ in self._fields_)


class PredictionZCU(ct.Structure):
    """ Prediction (ZCU) - This class defines a C-like struct """
    _fields_ = [
        ("top_power", ct.c_float),
        ("bottom_power", ct.c_float),
        ("time", ct.c_float)
    ]


class PredictionPYNQ(ct.Structure):
    """ Prediction (PYNQ) - This class defines a C-like struct """
    _fields_ = [
        ("power", ct.c_float),
        ("time", ct.c_float)
    ]


class MetricsZCU(ct.Structure):
    """ Errro Metrics (ZCU) - This class defines a C-like struct """
    _fields_ = [
        ("ps_power_error", ct.c_float),
        ("pl_power_error", ct.c_float),
        ("time_error", ct.c_float)
    ]


class MetricsPYNQ(ct.Structure):
    """ Errro Metrics (PYNQ) - This class defines a C-like struct """
    _fields_ = [
        ("power_error", ct.c_float),
        ("time_error", ct.c_float)
    ]


def training_thread_file_func(online_models,
                              tcp_socket,
                              lock,
                              board,
                              cpu_usage):
    """Train models at run-time.
    (online data stored in disk-backed files)
    """

    # Wait for the client to connect via socket
    tcp_socket.wait_connection()
    print("[{:^19}] TCP socket connected.".format("Training Thread (f)"))

    # Set the path to the online.bin and traces files
    output_data_path = "../outputs"
    traces_data_path = "../traces"

    # Useful local variables
    t_interv = 0
    notification = 1
    num_measurements = 0
    i = 0
    is_training = True  # True means training, False means testing

    # Keep processsing files undefinatelly
    # (test with finite number of iterations)
    while (notification != 0):

        notification = tcp_socket.recv_data()
        print("notification: {}".format(notification))
        print(
            "[{:^19}]".format("Training Thread (f)"),
            datetime.now(timezone.utc)
        )

        # Unpack the received binary data into an integer value
        received_data = struct.unpack("I", notification)[0]

        # Check is the message is for training or testing (MSB)
        # 1 when training; 0 when testing
        is_training = True if received_data & (1 << 31) != 0 else False

        # Get the number of measurements (31 least significant bits)
        num_measurements = received_data & ~(1 << 31)

        if is_training:
            print("We are Training!")
        else:
            print("We are Testing!")
            # Generate tmp metrics
            if board == "ZCU":
                test_metrics = [
                    river.metrics.RMSE(),
                    river.metrics.RMSE(),
                    river.metrics.RMSE()
                    ]
            elif board == "PYNQ":
                test_metrics = [
                    river.metrics.RMSE(),
                    river.metrics.RMSE()
                    ]

        print("num_measurements: {}".format(num_measurements))
        if num_measurements == 0:
            break
        # Get number of measures to train with
        # num_measurements = int(notification)

        if i == 0:
            # Time measurement logic
            t_start = time.time()
        for iter in range(num_measurements):

            t_inter0 = time.time()

            i += 1

            # Generate the next online_info files path
            curr_output_path = os.path.join(output_data_path,
                                            "online_{}.bin".format(i-1))
            curr_power_path = os.path.join(traces_data_path,
                                           "CON_{}.BIN".format(i-1))
            curr_traces_path = os.path.join(traces_data_path,
                                            "SIG_{}.BIN".format(i-1))

            # Generate observations for this particular online_info.bin file
            generated_obs = \
                odp.generate_observations_from_online_data(
                    curr_output_path,
                    curr_power_path,
                    curr_traces_path,
                    board,
                    cpu_usage
                )

            # Check if this monitoring windows contains observations
            # In case there are no observations just move to next window
            if len(generated_obs) < 1:
                # print("[{:^19}] No useful obs".format("Training Thread (f)"))
                t_inter1 = time.time()
                t_interv += t_inter1-t_inter0
                continue

            # Create dataframe just for this online_data file
            df = odp.generate_dataframe_from_observations(
                generated_obs,
                board,
                cpu_usage
            )

            # DEBUG: Checking with observations are generated
            # print(curr_output_path)
            # print(traces_path)
            # print(curr_traces_file)
            # print(df)

            train_number_raw_observations = len(df)
            # remove nan (happens due to not using mutex for cpu_usage)
            df = df.dropna()

            train_number_observations = len(df)

            print("Train NaN rows: {}".format(
                train_number_raw_observations - train_number_observations
                )
            )

            # Different behaviour depending if is training or testing
            if is_training:
                # Learn batch with the online models
                online_models.train_s(df, lock)
            else:
                # Test batch with the online models
                test_metrics = online_models.test_s(df, lock, test_metrics)

            if board == "ZCU":
                # Different behaviour depending if is training or testing
                if is_training:
                    tmp_top_metric, tmp_bottom_metric, tmp_time_metric = \
                        online_models.get_metrics()
                else:
                    tmp_top_metric, tmp_bottom_metric, tmp_time_metric = \
                        test_metrics

                # Generate c-like structure containing the model error metrics
                metrics_to_send = MetricsZCU(
                    tmp_top_metric,
                    tmp_bottom_metric,
                    tmp_time_metric
                )

                print(
                    "[{:^19}] Metrics: {} (top) | {} (bottom)"
                    " | {} (time)".format(
                        "Training Thread (f)",
                        tmp_top_metric,
                        tmp_bottom_metric,
                        tmp_time_metric
                    )
                )
            elif board == "PYNQ":
                # Different behaviour depending if is training or testing
                if is_training:
                    tmp_power_metric, tmp_time_metric = \
                        online_models.get_metrics()
                else:
                    tmp_power_metric, tmp_time_metric = test_metrics

                # Generate c-like structure containing the model error metrics
                metrics_to_send = MetricsPYNQ(
                    tmp_power_metric,
                    tmp_time_metric
                )

                print(
                    "[{:^19}] Metrics: {} (power) | {} (time)".format(
                        "Training Thread (f)",
                        tmp_power_metric,
                        tmp_time_metric
                    )
                )

            t_inter1 = time.time()
            t_interv += t_inter1-t_inter0

        # Send the metrics obtained via socket
        tcp_socket.send_data(metrics_to_send)

    # Time measurement logic
    t_end = time.time()

    # Close the socket
    tcp_socket.close_connection()

    # Print useful information
    if i > 0:  # Take care of division by zero
        print(
            "[{:^19}] Interval Elapsed Time (s):".format(
                "Training Thread (f)"),
            t_interv,
            (t_interv)/i
        )
        print(
            "[{:^19}] Total Elapsed Time (s):".format("Training Thread (f)"),
            t_end-t_start,
            (t_end-t_start)/i
        )
        print(
            "[{:^19}] Number of trainings: {}".format("Training Thread (f)", i)
        )

    else:
        print("[{:^19}] No processing was made".format("Training Thread (f)"))

    if board == "ZCU":
        tmp_top_metric, tmp_bottom_metric, tmp_time_metric = \
            online_models.get_metrics()
        print(
            "[{:^19}] Training Metrics: {} (top) | {} (bottom) "
            "| {} (time)".format(
                "Training Thread (f)",
                tmp_top_metric,
                tmp_bottom_metric,
                tmp_time_metric
            )
        )
    elif board == "PYNQ":
        tmp_power_metric, tmp_time_metric = online_models.get_metrics()
        print(
            "[{:^19}] Training Metrics: {} (top) | {} (time)".format(
                "Training Thread (f)",
                tmp_power_metric,
                tmp_time_metric
            )
        )

    print("[{:^19}] Thread terminated.".format("Training Thread (f)"))


def training_thread_ram_func(online_models,
                             tcp_socket,
                             lock,
                             board,
                             cpu_usage):
    """Train models at run-time. (or test is the c code wants to check metrics)
    (online_data stored in ram-backed mmap'ed files)
    """

    # Wait for the client to connect via socket
    tcp_socket.wait_connection()
    print("[{:^19}] TCP socket connected.".format("Training Thread (r)"))

    # Get number of training iterations
    notification = tcp_socket.recv_data()
    print("notification: {}".format(notification))
    print("[{:^19}]".format("Training Thread (r)"),
          datetime.now(timezone.utc))
    # Unpack the received binary data into an integer value
    number_iterations = struct.unpack("I", notification)[0]
    print("num_measurements: {}".format(number_iterations))
    # Send ack indicating ack of the number_iterations
    tcp_socket.send_data(b'1')
    print("Sent number_iterations ack to the c program")

    # Ping-Pong/Execution-modes buffers configuration variables
    # (maybe makes sense to allow some kind of configurability to the user)
    SHM_ONLINE_FILE = "online"
    SHM_POWER_FILE = "power"
    SHM_TRACES_FILE = "traces"

    SHM_ONLINE_REGION_SIZE = 2*1024   # Bytes
    SHM_POWER_REGION_SIZE = 525*1024  # Bytes
    SHM_TRACES_REGION_SIZE = 20*1024  # Bytes

    # Create the ping-pong buffers
    if number_iterations == 1:
        buffers = ppb.PingPongBuffers(SHM_ONLINE_FILE,
                                      SHM_ONLINE_REGION_SIZE,
                                      SHM_POWER_FILE,
                                      SHM_POWER_REGION_SIZE,
                                      SHM_TRACES_FILE,
                                      SHM_TRACES_REGION_SIZE,
                                      create=True,
                                      unregister=True)
    elif number_iterations > 1:
        buffers = emb.ExecutionModesBuffers(SHM_ONLINE_FILE,
                                            SHM_ONLINE_REGION_SIZE,
                                            SHM_POWER_FILE,
                                            SHM_POWER_REGION_SIZE,
                                            SHM_TRACES_FILE,
                                            SHM_TRACES_REGION_SIZE,
                                            number_iterations,
                                            create=True,
                                            unregister=True)
    else:
        raise Exception("Number of iterations cannot be less than 1.")

    # Useful local variables
    t_interv = 0
    notification = 1
    i = 0
    num_useless_obs = 0
    is_training = True  # True means training, False means testing

    # Debug
    t_copy_buf = 0
    t_gen_obs = 0
    t_toggle = 0
    t_gen_dataframe = 0
    t_print_df = 0
    t_train = 0
    t_metrics = 0

    # Keep processsing files undefinatelly
    # (test with finite number of iterations)
    while (notification != 0):

        notification = tcp_socket.recv_data()
        print("notification: {}".format(notification))
        print("[{:^19}]".format("Training Thread (r)"),
              datetime.now(timezone.utc))

        # Unpack the received binary data into an integer value
        received_data = struct.unpack("I", notification)[0]

        # Check is the message is for training or testing (MSB)
        # 1 when training; 0 when testing
        is_training = True if received_data & (1 << 31) != 0 else False

        # Get the number of measurements (31 least significant bits)
        num_measurements = received_data & ~(1 << 31)

        if is_training:
            print("We are Training!")
        else:
            print("We are Testing!")
            # Generate tmp metrics
            if board == "ZCU":
                test_metrics = [
                    river.metrics.RMSE(),
                    river.metrics.RMSE(),
                    river.metrics.RMSE(),
                    ]
            elif board == "PYNQ":
                test_metrics = [
                    river.metrics.RMSE(),
                    river.metrics.RMSE(),
                    river.metrics.RMSE(),
                    ]

        print("num_measurements: {}".format(num_measurements))
        if num_measurements == 0:
            break
        # Get number of measurements to train with
        # num_measurements = int(notification)

        if i == 0:
            # Time measurement logic
            t_start = time.time()
        for iter in range(num_measurements):

            t_inter0 = time.time()

            i += 1

            # Get last int (it contains the size of the actual data)
            online_size = int.from_bytes(buffers.online[-4:], sys.byteorder)
            power_size = int.from_bytes(buffers.power[-4:], sys.byteorder)
            traces_size = int.from_bytes(buffers.traces[-4:], sys.byteorder)

            # print("Write to files")
            # with open("test_ram/CON_{}.BIN".format(iter), "wb") as b_file:
            #     # Write bytes to file
            #     binary_file.write(buffers.power[:power_size])
            # with open("test_ram/SIG_{}.BIN".format(iter), "wb") as b_file:
            #     # Write bytes to file
            #     binary_file.write(buffers.traces[:traces_size])
            # with open("test_ram/online_{}.bin".format(iter), "wb") as b_file:
            #     # Write bytes to file
            #     binary_file.write(buffers.online[:online_size])

            # print(online_size)
            # print(power_size)
            # print(traces_size)

            t_inter1 = time.time()

            # Generate observations for this particular online_info.bin file
            generated_obs = \
                odp.generate_observations_from_online_data(
                    buffers.online[:online_size],
                    buffers.power[:power_size],
                    buffers.traces[:traces_size],
                    board,
                    cpu_usage)

            t_inter2 = time.time()

            # print(generated_obs)

            # Toggle current buffers
            buffers.toggle()

            t_inter3 = time.time()

            # print(generated_obs)

            # Check if this monitoring windows contains observations
            # In case there are no observations just move to next window
            if len(generated_obs) < 1:
                print("[{:^19}] No useful obs".format("Training Thread (r)"))
                t_inter7 = time.time()
                t_interv += t_inter7-t_inter0
                num_useless_obs += 1

                # Debug
                t_copy_buf += t_inter1 - t_inter0
                t_gen_obs += t_inter2 - t_inter1
                t_toggle += t_inter3 - t_inter2
                continue

            # Create dataframe just for this online_data file
            df = odp.generate_dataframe_from_observations(
                generated_obs,
                board,
                cpu_usage)

            t_inter4 = time.time()

            # DEBUG: Checking with observations are generated
            # print(curr_output_path)
            # print(traces_path)
            # print(curr_traces_file)
            # print(df)

            t_inter5 = time.time()

            train_number_raw_observations = len(df)
            # remove nan (happens due to not using mutex for cpu_usage)
            df = df.dropna()

            train_number_observations = len(df)

            print("Train NaN rows: {}".format(
                train_number_raw_observations - train_number_observations
                )
            )

            if len(df.index) < 1:
                num_useless_obs += 1
                continue

            # Different behaviour depending if is training or testing
            if is_training:
                # Learn batch with the online models
                online_models.train_s(df, lock)
            else:
                # Test batch with the online models
                test_metrics = online_models.test_s(df, lock, test_metrics)

            t_inter6 = time.time()

            if board == "ZCU":
                # Different behaviour depending if is training or testing
                if is_training:
                    tmp_top_metric, tmp_bottom_metric, tmp_time_metric = \
                        online_models.get_metrics()
                else:
                    tmp_top_metric, tmp_bottom_metric, tmp_time_metric = \
                        test_metrics

                # Generate c-like structure containing the model error metrics
                metrics_to_send = MetricsZCU(
                    tmp_top_metric.get(),
                    tmp_bottom_metric.get(),
                    tmp_time_metric.get()
                )

                print(
                    "[{:^19}] Metrics: {} (top) | {} (bottom)"
                    " | {} (time)".format(
                        "Training Thread (r)",
                        tmp_top_metric,
                        tmp_bottom_metric,
                        tmp_time_metric
                    )
                )
            elif board == "PYNQ":
                # Different behaviour depending if is training or testing
                if is_training:
                    tmp_power_metric, tmp_time_metric = \
                        online_models.get_metrics()
                else:
                    tmp_power_metric, tmp_time_metric = test_metrics

                # Generate c-like structure containing the model error metrics
                metrics_to_send = MetricsPYNQ(
                    tmp_power_metric.get(),
                    tmp_time_metric.get()
                )

                print(
                    "[{:^19}] Metrics: {} (power) | {} (time)".format(
                        "Training Thread (r)",
                        tmp_power_metric,
                        tmp_time_metric
                    )
                )

            t_inter7 = time.time()
            t_interv += t_inter7-t_inter0

            # Debug
            t_copy_buf += t_inter1 - t_inter0
            t_gen_obs += t_inter2 - t_inter1
            t_toggle += t_inter3 - t_inter2
            t_gen_dataframe += t_inter4 - t_inter3
            t_print_df += t_inter5 - t_inter4
            t_train += t_inter6 - t_inter5
            t_metrics += t_inter7 - t_inter6

        # Send the metrics obtained via socket
        tcp_socket.send_data(metrics_to_send)

    # Time measurement logic
    t_end = time.time()

    # Close the socket
    tcp_socket.close_connection()

    # Print useful information
    if i > 0:  # Take care of division by zero
        print("[{:^19}] Number of trainings: {}".format("Training Thread (r)",
                                                        i))
        print(
            "[{:^19}] Total Elapsed Time (s)   : (total) {:016.9f}"
            " | (ratio) {:012.9f}".format(
                "Training Thread (r)",
                t_end - t_start,
                (t_end - t_start) / i)
        )
        print(
            "[{:^19}] Interval Elapsed Time (s): (total) {:016.9f}"
            " | (ratio) {:012.9f}".format(
                "Training Thread (r)",
                t_interv,
                t_interv / i)
        )
        # Debug
        print(
            "[{:^19}] t_copy_buf (s)           : (total) {:016.9f}"
            " | (ratio) {:012.9f}".format(
                "Training Thread (r)",
                t_copy_buf,
                t_copy_buf / i)
        )
        print(
            "[{:^19}] t_gen_obs (s)            : (total) {:016.9f}"
            " | (ratio) {:012.9f}".format(
                "Training Thread (r)",
                t_gen_obs,
                t_gen_obs / i)
        )
        print(
            "[{:^19}] t_toggle (s)             : (total) {:016.9f}"
            " | (ratio) {:012.9f}".format(
                "Training Thread (r)",
                t_toggle,
                t_toggle / i)
        )
        print(
            "[{:^19}] t_gen_dataframe (s)      : (total) {:016.9f}"
            " | (ratio) {:012.9f}".format(
                "Training Thread (r)",
                t_gen_dataframe,
                t_gen_dataframe / (i - num_useless_obs))
        )
        print(
            "[{:^19}] t_print_df (s)           : (total) {:016.9f}"
            " | (ratio) {:012.9f}".format(
                "Training Thread (r)",
                t_print_df,
                t_print_df / (i - num_useless_obs))
        )
        print(
            "[{:^19}] t_train (s)              : (total) {:016.9f}"
            " | (ratio) {:012.9f}".format(
                "Training Thread (r)",
                t_train,
                t_train / (i - num_useless_obs))
        )
        print(
            "[{:^19}] t_metrics (s)            : (total) {:016.9f}"
            " | (ratio) {:012.9f}".format(
                "Training Thread (r)",
                t_metrics,
                t_metrics / (i - num_useless_obs))
        )
    else:
        print("[{:^19}] No processing was made".format("Training Thread (r)"))

    if board == "ZCU":
        tmp_top_metric, tmp_bottom_metric, tmp_time_metric = \
            online_models.get_metrics()
        print(
            "[{:^19}] Training Metrics: {} (top) | {} (bottom) "
            "| {} (time)".format(
                "Training Thread (r)",
                tmp_top_metric,
                tmp_bottom_metric,
                tmp_time_metric
            )
        )
    elif board == "PYNQ":
        tmp_power_metric, tmp_time_metric = online_models.get_metrics()
        print(
            "[{:^19}] Training Metrics: {} (power) | {} (time)".format(
                "Training Thread (r)",
                tmp_power_metric,
                tmp_time_metric
            )
        )

    # Close shared memories (unlink if desired)
    buffers.clean(unlink=True)

    print("[{:^19}] Thread terminated.".format("Training Thread (r)"))


def prediction_thread_func(online_models,
                           tcp_socket,
                           lock,
                           board,
                           cpu_usage):
    """Use models to make a prediction at run-time.
    """

    print("[{:^19}] In".format("Prediction Thread"))

    # Wait for the client to connect via socket
    tcp_socket.wait_connection()
    print("[{:^19}] TCP socket connected.".format("Prediction Thread"))

    t_interv = 0
    buff = b'1'
    i = 0
    # Keep making predictions
    while (buff != b'0'):

        buff = tcp_socket.recv_data()
        print("buffer: {}".format(buff))
        print(
            "[{:^19}]".format("Prediction Thread"),
            datetime.now(timezone.utc)
        )

        if i == 0:
            # Time measurement logic
            t_start = time.time()

        if buff != b'0':

            t_inter0 = time.time()
            # t0 = time.time()

            # Convert from c-like struct to python
            if cpu_usage:
                features_c = FeatureswCPUUsage.from_buffer_copy(buff)
            else:
                features_c = FeatureswoCPUUsage.from_buffer_copy(buff)
            features = features_c.get_dict()

            # DEBUG: print prediction request
            print(features)

            # Make one prediction with the models

            # t1 = time.time()

            if board == "ZCU":
                top_power_prediction, \
                    bottom_power_prediction, \
                    time_prediction = \
                    online_models.predict_one_s(features, lock)

                # t2 = time.time()

                # Generate c-like structure for predictions
                prediction = PredictionZCU(
                    top_power_prediction,
                    bottom_power_prediction,
                    time_prediction
                )
            elif board == "PYNQ":
                power_prediction, \
                    time_prediction = \
                    online_models.predict_one_s(features, lock)

                # t2 = time.time()

                # Generate c-like structure for predictions
                prediction = PredictionPYNQ(
                    power_prediction,
                    time_prediction
                )

            # t3 = time.time()

            # Sed the prediction via socket
            tcp_socket.send_data(prediction)

            # t4 = time.time()

            # print("[{:^19}] buff:".format("Prediction Thread"), buff)
            # print(
            #   "[{:^19}] features_c:".format("Prediction Thread"),
            #   features_c
            # )
            # print("[{:^19}] features:".format("Prediction Thread"), features)
            # print("[{:^19}] top: {} | bot: {} | time: {}".format(
            #     "Prediction Thread",
            #     top_power_prediction,
            #     bottom_power_prediction,
            #     time_prediction
            # ))
            # print(
            #   "[{:^19}] prediction".format("Prediction Thread"),
            #   prediction
            # )
            # print(
            #     "[{:^19}] prediction_thread: {} (conv_featues) |"
            #     "{} (predict) | {} (conv_pred) | {} (send_pred)".format(
            #         "Prediction Thread",
            #         t1-t0,
            #         t2-t1,
            #         t3-t2,
            #         t4-t3
            #     )
            # )

            # Time measurement logic
            t_inter1 = time.time()
            t_interv += t_inter1-t_inter0

            # Increment the counter
            i += 1

            # test
            if board == "ZCU":
                tmp_top_metric, tmp_bottom_metric, tmp_time_metric = \
                    online_models.get_metrics()
                print(
                    "[{:^19}] Training Metrics: {} (top) | {} (bottom)"
                    " | {} (time)".format(
                        "Prediction Thread",
                        tmp_top_metric,
                        tmp_bottom_metric,
                        tmp_time_metric
                    )
                )
            elif board == "PYNQ":
                tmp_power_metric, tmp_time_metric = online_models.get_metrics()
                print(
                    "[{:^19}] Training Metrics: {} (power) | {} (time)".format(
                        "Prediction Thread",
                        tmp_power_metric,
                        tmp_time_metric
                    )
                )

    # Close the socket
    tcp_socket.close_connection()
    # Time measurement logic
    t_end = time.time()

    # Print useful information
    if i > 0:  # Take care of division by zero
        print(
            "[{:^19}] Interval Elapsed Time (s):".format("Prediction Thread"),
            t_interv,
            (t_interv)/i
        )
        print(
            "[{:^19}] Total Elapsed Time (s):".format("Prediction Thread"),
            t_end-t_start,
            (t_end-t_start)/i
        )
        print("[{:^19}] Number of predictions: {}".format(
            "Prediction Thread",
            i)
        )

    else:
        print("[[{:^19}] No predictions was made".format("Prediction Thread"))

    print("[{:^19}] Thread terminated.".format("Prediction Thread"))


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()

    # Indicate where are the traces and online data stored
    parser.add_argument("memory",
                        choices=["rom", "ram"],
                        help="Type of memory where online data is stored")

    # Indicate which board has been used
    parser.add_argument("board",
                        choices=["ZCU", "PYNQ"],
                        help="Type of board used")

    # Indicate if cpu usage or not (default is true)
    parser.add_argument('--no-cpu-usage', action='store_true')

    args = parser.parse_args(sys.argv[1:])

    # Variable indicating cpu usage
    cpu_usage_flag = not args.no_cpu_usage

    print("CPU usage: {}".format(cpu_usage_flag))

    # Select the training thread function based on the type of memory used
    if args.memory == "rom":
        training_thread_func = training_thread_file_func
    else:
        training_thread_func = training_thread_ram_func

    # Select the prediction and metrics structure based on the board used
    if args.board == "ZCU":
        Prediction = PredictionZCU
        Metrics = MetricsZCU
    else:
        Prediction = PredictionPYNQ
        Metrics = MetricsPYNQ

    # Select the appropriate features structure based on cpu usage availability
    if cpu_usage_flag:
        Features = FeatureswCPUUsage
    else:
        Features = FeatureswoCPUUsage

    # DEBUG: no dataframe printing column limit
    pd.set_option('display.max_columns', None)

    # Create the UDP server socket used to trigger the processing of an
    # online_info.bin file

    tcp_train_socket = ipc.ServerSocketTCP("my_training_socket",
                                           struct.calcsize("I"),
                                           ct.sizeof(Metrics))
    print("[{:^19}] TCP socket created.".format("Main Thread"))

    # Create TCP server socket used to trigger the models' predictoin process
    tcp_prediction_socket = ipc.ServerSocketTCP("my_prediction_socket",
                                                ct.sizeof(Features),
                                                ct.sizeof(Prediction))
    print("[{:^19}] TCP socket created.".format("Main Thread"))

    # Create thread safe lock
    lock = Lock()

    # Initialize the online models
    online_models = om.OnlineModels(board=args.board)
    print("[{:^19}] Online Models have been successfully initialized.".format(
        "Main Thread")
    )

    # Create the training and prediction threads
    training_thread = Thread(
        target=training_thread_func,
        args=(online_models,
              tcp_train_socket,
              lock,
              args.board,
              cpu_usage_flag)
    )
    prediction_thread = Thread(
        target=prediction_thread_func,
        args=(online_models,
              tcp_prediction_socket,
              lock,
              args.board,
              cpu_usage_flag)
    )
    print("[{:^19}] Both threads have been successfully created.".format(
        "Main Thread")
    )

    # Start both threads
    training_thread.start()
    prediction_thread.start()
    print("[{:^19}] Both threads have been successfully started.".format(
        "Main Thread")
    )

    # Ensure all threads have finished
    training_thread.join()
    prediction_thread.join()
    print("[{:^19}] Both threads have successfully finished.".format(
        "Main Thread")
    )

    # Set observations file path
    # (there because is executed from ssh without permission in folder outputs)
    # Solvable by changing the ownership of that folder
    # observations_path = "/home/artico3/runtime_observations.pkl"

    # Save the dataframe in a file
    # df.to_pickle(observations_path)
