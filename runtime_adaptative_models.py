#!/usr/bin/env python3

"""
Run-Time Training and Prediction with Online Models

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : December 2023
Description : This script:
              - Processes power and performance traces received from an
                independent c-code process via disk-backed files or ram-backed
                memory-mapped regions (up to the user).
              - Trains power (pl and ps) and performance online models on
                demand from commands send via a socket from the independent
                c-code process.
              - Predicts with the trained models for the features receive from
                the independent c-code process via another socket.
              - Models are synchronized, training and testing simultaneously
              All is done at run-time with concurrent threads for training the
              models and predicting with them.

"""

import sys
import os
import argparse
import time
import struct
import threading
from datetime import datetime, timezone
# from multiprocessing import Process
import ctypes as ct
import pandas as pd
import river
import pickle

from incremental_learning import online_data_processing as odp
from incremental_learning import inter_process_communication as ipc
from incremental_learning import online_models as om
from incremental_learning import ping_pong_buffers as ppb
from incremental_learning import execution_modes_buffers as emb


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


def training_thread_file_func(online_models, tcp_socket, board, cpu_usage):
    """Train models at run-time. (online data stored in disk-backed files)"""

    # Wait for the client to connect via socket
    tcp_socket.wait_connection()
    print(f"[{'Training Thread (f)':^19}] TCP socket connected.")

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
    while notification != 0:

        notification = tcp_socket.recv_data()
        print(f"notification: {notification}")
        print(f"[{'Training Thread (f)':^19}] {datetime.now(timezone.utc)}")

        # Unpack the received binary data into an integer value
        received_data = struct.unpack("i", notification)[0]

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

        print(f"num_measurements: {num_measurements}")
        if num_measurements == 0:
            break
        # Get number of measures to train with
        # num_measurements = int(notification)

        if i == 0:
            # Time measurement logic
            t_start = time.time()
        for _ in range(num_measurements):

            t_inter0 = time.time()

            i += 1

            # Generate the next online_info files path
            curr_output_path = os.path.join(output_data_path, f"online_{i-1}.bin")
            curr_power_path = os.path.join(traces_data_path, f"CON_{i-1}.BIN")
            curr_traces_path = os.path.join(traces_data_path, f"SIG_{i-1}.BIN")

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
            dataframe = odp.generate_dataframe_from_observations(
                generated_obs,
                board,
                cpu_usage
            )

            # DEBUG: Checking with observations are generated
            # print(curr_output_path)
            # print(traces_path)
            # print(curr_traces_file)
            # print(dataframe)

            train_number_raw_observations = len(dataframe)
            # remove nan (happens due to not using mutex for cpu_usage)
            dataframe = dataframe.dropna()

            train_number_observations = len(dataframe)

            print(f"Train NaN rows: {train_number_raw_observations - train_number_observations}")

            # Different behaviour depending if is training or testing
            if is_training:
                # Learn batch with the online models
                online_models.train_s(dataframe)
            else:
                # Test batch with the online models
                test_metrics = online_models.test_s(dataframe, test_metrics)

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
                    f"[{'Training Thread (f)':^19}] Metrics: {tmp_top_metric} (top) | "
                    f"{tmp_bottom_metric} (bottom) | {tmp_time_metric} (time)"
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
                    f"[{'Training Thread (f)':^19}] Metrics: {tmp_power_metric} (power) | "
                    f"{tmp_time_metric} (time)"
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
        print(f"[{'Training Thread (f)':^19}] Interval Elapsed Time (s):", t_interv, (t_interv)/i)
        print(
            f"[{'Training Thread (f)':^19}] Total Elapsed Time (s):",
            t_end-t_start,
            (t_end-t_start)/i
            )
        print(f"[{'Training Thread (f)':^19}] Number of trainings: {i}")

    else:
        print(f"[{'Training Thread (f)':^19}] No processing was made")

    if board == "ZCU":
        tmp_top_metric, tmp_bottom_metric, tmp_time_metric = \
            online_models.get_metrics()
        print(
            f"[{'Training Thread (f)':^19}] Training Metrics: {tmp_top_metric} (top) | "
            f"{tmp_bottom_metric} (bottom) | {tmp_time_metric} (time)"
            )
    elif board == "PYNQ":
        tmp_power_metric, tmp_time_metric = online_models.get_metrics()
        print(
            f"[{'Training Thread (f)':^19}] Training Metrics: {tmp_power_metric} (top) | "
            f"{tmp_time_metric} (time)"
            )

    print(f"[{'Training Thread (f)':^19}] Thread terminated.")


def training_thread_ram_func(online_models, tcp_socket, board, cpu_usage):
    """Train models at run-time. (or test is the c code wants to check metrics)
    (online_data stored in ram-backed mmap'ed files)
    """

    # Wait for the client to connect via socket
    tcp_socket.wait_connection()
    print(f"[{'Training Thread (r)':^19}] TCP socket connected.")

    # Get number of training iterations
    notification = tcp_socket.recv_data()
    print(f"notification: {notification}")
    print(f"[{'Training Thread (r)':^19}]", datetime.now(timezone.utc))
    # Unpack the received binary data into an integer value
    number_iterations = struct.unpack("i", notification)[0]
    print(f"num_measurements: {number_iterations}")

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

    # Send ack indicating ack of the number_iterations
    tcp_socket.send_data(b'1')
    print("Sent number_iterations ack to the c program")

    # Useful local variables
    t_interv = 0
    notification = 1
    i = 0
    num_useless_obs = 0

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
    # Local Online Models Manager variables
    iteration = 0
    next_operation_mode = "train"
    wait_obs = 0

    # TODO: Remove. Used to store workload phaces
    workloads_buffer_index = [[0,]]
    workloads_obs_index = [[0,]]

    # TODO: Test. Variable to store info for later generate temporal axis
    # Formato: lista de listas. Cada sub lista contendrá la info de una etapa de train/test
    #          (idle no se recoje pero son el resto) necesaria para regenerar la gráfica
    # [num_measurements, start_time_train/test, start_observation_index, end_time_train/test, end_observation_index]
    # Como tenemos una lista con el error para cada observación y el indice de estas con la sublista anterior
    # podemos identificar en qué momento se genera cada obs (distribuyendolas uniformement en la etapa medida+train/test)
    # * Es cierto qeu el número de obs en idle puede no ser el real pues se hace una aproximación en el setup, pero como
    #   tenemos el tiempo justo del final de la etapa train/test anterior y el tiempo inicial de la siguiente sabemos ubicarlo
    temporal_data = []

    # TODO: Remove. Used to check ratio of observations per window
    total_observations = 0
    total_windows = 0
    while notification != 0:

        notification = tcp_socket.recv_data()
        print(f"notification: {notification}")
        print(f"[{'Training Thread (r)':^19}]", datetime.now(timezone.utc))

        # Unpack the received binary data into an integer value
        received_data = struct.unpack("i", notification)[0]

        # Get the number of measurements
        num_measurements = received_data

        print(f"num_measurements: {num_measurements}")
        if num_measurements < 0:
            # Setup has indicated next obs come from new workload
            print(f"Next Workload at buffer: {i} (iteration: {iteration})")
            # Mark last workload buffer and obs index and append new list to mark next workload
            workloads_buffer_index[-1].append(i-1)
            workloads_buffer_index.append([i-1])
            workloads_obs_index[-1].append(iteration-1)
            workloads_obs_index.append([iteration-1])
            continue
        elif num_measurements == 0:
            # Setup indicates end of execution
            # Mark last workload end index
            workloads_buffer_index[-1].append(i)
            workloads_obs_index[-1].append(iteration)
            break

        if i == 0:
            # Time measurement logic
            t_start = time.time()

        # Train/test from each measurement
        # TODO:Test para generar eje temporal
        temporal_data.append([])
        temporal_data[-1].append(num_measurements)
        temporal_data[-1].append(time.time())
        temporal_data[-1].append(iteration)
        for iter in range(num_measurements):

            t_inter0 = time.time()

            i += 1

            # Get last int (it contains the size of the actual data)
            online_size = int.from_bytes(buffers.online[-4:], sys.byteorder)
            power_size = int.from_bytes(buffers.power[-4:], sys.byteorder)
            traces_size = int.from_bytes(buffers.traces[-4:], sys.byteorder)

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

            # Toggle current buffers
            buffers.toggle()

            t_inter3 = time.time()

            # Check if this monitoring windows contains observations
            # In case there are no observations just move to next window
            if len(generated_obs) < 1:
                # print(f"[{'Training Thread (r)':^19}] No useful obs")
                t_inter5 = time.time()
                t_interv += t_inter5-t_inter0
                num_useless_obs += 1

                # Debug
                t_copy_buf += t_inter1 - t_inter0
                t_gen_obs += t_inter2 - t_inter1
                t_toggle += t_inter3 - t_inter2
                continue

            # Create dataframe just for this online_data file
            dataframe = odp.generate_dataframe_from_observations(
                generated_obs,
                board,
                cpu_usage)

            t_inter4 = time.time()

            # train_number_raw_observations = len(dataframe)
            # remove nan (happens due to not using mutex for cpu_usage)
            dataframe = dataframe.dropna()


            # TODO: Remove. Used to check ratio of observations per window
            total_windows += 1
            total_observations += len(dataframe)

            # print(f"Train NaN rows: {train_number_raw_observations - train_number_observations}")

            # Skip useless observations
            if len(dataframe.index) < 1:
                num_useless_obs += 1
                continue

            # Let the models manager decide whether train or test
            # TODO: Quizá es más eficiente no entrenar con cada iteracion, sino concatenar los
            #       dataframes generados y entrenar una vez hayan llegado todos
            iteration, next_operation_mode, wait_obs = online_models.update_models(
                dataframe,
                iteration
                )

            t_inter5 = time.time()

            t_interv += t_inter5-t_inter0

            # Debug
            t_copy_buf += t_inter1 - t_inter0
            t_gen_obs += t_inter2 - t_inter1
            t_toggle += t_inter3 - t_inter2
            t_gen_dataframe += t_inter4 - t_inter3
            t_train += t_inter5 - t_inter4

            # TODO: esto podría cambiar si en lugar de entrenar en cada iteración entrenamos al
            #       final, pero es cierto que al hacerlo cada cada iteración ahorraríamos tiempo
            #       de procesado de las iteraciones innecesarias
            if next_operation_mode == "idle":
                print(f"Reached Idle before processing all measurements: ({iter}/{num_measurements}")
                break

        # TODO: variables para añadir eje temporal
        temporal_data[-1].append(time.time())
        if next_operation_mode == "idle":
            # In case we are in idle in next operation mode we have to substract the wait_obs since
            # inside the update_models the training monitor increases the iteration value by
            # wait_obs before it happens.
            # So to ensure the postprocessing of the temporal data is properly applied we need to
            # substract that
            temporal_data[-1].append(iteration - wait_obs - 1)
        else:
            temporal_data[-1].append(iteration - 1)  # Since update_models increases the iteration

        # TODO: Lo de entrenar of test se quita porque ahora lo gestion el Training Monitor
        # Send the metrics obtained via socket
        # tcp_socket.send_data(metrics_to_send)
        ack_value = wait_obs if next_operation_mode == "idle" else 0
        ack_bytes = struct.pack("i", ack_value)
        tcp_socket.send_data(ack_bytes)

        print(f"[Pos-train]: iter: {iteration} | mode: {next_operation_mode} | wait_obs: {wait_obs}")

        # Print useful information
        if i > 0:  # Take care of division by zero
            print(f"[{'Training Thread (r)':^19}] Number of trainings: {i}")

            print(
                f"[{'Training Thread (r)':^19}] Interval Elapsed Time (s): "
                f"(total) {t_interv:016.9f} | (ratio) {t_interv / i:012.9f}"
                )

            # Debug
            print(
                f"[{'Training Thread (r)':^19}] t_copy_buf (s)           : "
                f"(total) {t_copy_buf:016.9f} | (ratio) {t_copy_buf / i:012.9f}"
                )
            print(
                f"[{'Training Thread (r)':^19}] t_gen_obs (s)            : "
                f"(total) {t_gen_obs:016.9f} | (ratio) {t_gen_obs / i:012.9f}"
                )
            print(
                f"[{'Training Thread (r)':^19}] t_toggle (s)             : "
                f"(total) {t_toggle:016.9f} | (ratio) {t_toggle / i:012.9f}"
                )
            print(
                f"[{'Training Thread (r)':^19}] t_gen_dataframe (s)      : "
                f"(total) {t_gen_dataframe:016.9f} | "
                f"(ratio) {t_gen_dataframe / (i - num_useless_obs):012.9f}"
                )
            print(
                f"[{'Training Thread (r)':^19}] t_train (s)              : "
                f"(total) {t_train:016.9f} | (ratio) {t_train / (i - num_useless_obs):012.9f}"
                )

            # TODO: Send back to the setup what to do
            print(
                f"{'Training Thread (r)':^19}] Iteration: {iteration} | Mode: {next_operation_mode} | "
                f"Wait Obs: {wait_obs}"
                )

    # Time measurement logic
    t_end = time.time()

    # Close the socket
    tcp_socket.close_connection()

    # Print useful information
    if i > 0:  # Take care of division by zero
        print(f"[{'Training Thread (r)':^19}] Number of trainings: {i}")

        print(
            f"[{'Training Thread (r)':^19}] Total Elapsed Time (s):    "
            f"(total) {t_end - t_start:016.9f} | (ratio) {(t_end - t_start) / i:012.9f}"
            )
        print(
            f"[{'Training Thread (r)':^19}] Interval Elapsed Time (s): "
            f"(total) {t_interv:016.9f} | (ratio) {t_interv / i:012.9f}"
            )

        # Debug
        print(
            f"[{'Training Thread (r)':^19}] t_copy_buf (s)           : "
            f"(total) {t_copy_buf:016.9f} | (ratio) {t_copy_buf / i:012.9f}"
            )
        print(
            f"[{'Training Thread (r)':^19}] t_gen_obs (s)            : "
            f"(total) {t_gen_obs:016.9f} | (ratio) {t_gen_obs / i:012.9f}"
            )
        print(
            f"[{'Training Thread (r)':^19}] t_toggle (s)             : "
            f"(total) {t_toggle:016.9f} | (ratio) {t_toggle / i:012.9f}"
            )
        print(
            f"[{'Training Thread (r)':^19}] t_gen_dataframe (s)      : "
            f"(total) {t_gen_dataframe:016.9f} | "
            f"(ratio) {t_gen_dataframe / (i - num_useless_obs):012.9f}"
            )
        print(
            f"[{'Training Thread (r)':^19}] t_train (s)              : "
            f"(total) {t_train:016.9f} | (ratio) {t_train / (i - num_useless_obs):012.9f}"
            )

        print(f"Total Windows: {total_windows} | Total Observations: {total_observations} | Ratio: {total_observations/total_windows}")

        # Print workload indexes
        print("Workloads buffer indexes", workloads_buffer_index)
        print("Workloads obs indexes", workloads_obs_index)
        # Print temporal data
        print("Temporal Data:", temporal_data)
    else:
        print(f"[{'Training Thread (r)':^19}] No processing was made")

    # TODO: Remove
    # TODO: Está triplicado, si nos cargamos 2 replicas?
    # When there are no more obs the system is either in train or test mode.
    # We need fill the last test/train_region list with the actual iteration
    for model in online_models._models:
        if model._training_monitor.operation_mode == "train":
            model._training_monitor.train_train_regions[-1].append(iteration-1)
        elif model._training_monitor.operation_mode == "test":
            model._training_monitor.test_test_regions[-1].append(iteration-1)
    ###############

    # Generar gráficas
    # online_models.print_training_monitor_info()

    # Guardar temporal data y modelos en fichero (online_models y temporal_data)
    # Create directory if it does not exit
    model_error_figures_dir = "./model_error_figures"
    if not os.path.exists(model_error_figures_dir):
        os.makedirs(model_error_figures_dir)

    # Ask the user for the figure name
    data_save_file_name = input("Give me name to save models object and temporal data "
                                f"(path:{model_error_figures_dir}/<name>.pkl): ")

    # Save the models training monitor
    with open(f"{model_error_figures_dir}/{data_save_file_name}_training_monitor.pkl", 'wb') as file:
        tmp_var = [model._training_monitor for model in online_models._models]
        pickle.dump(tmp_var, file)

    # Save the temporal data
    with open(f"{model_error_figures_dir}/{data_save_file_name}_temporal_data.pkl", 'wb') as file:
        pickle.dump(temporal_data, file)


    if board == "ZCU":
        tmp_top_metric, tmp_bottom_metric, tmp_time_metric = \
            online_models.get_metrics()
        print(
            f"[{'Training Thread (r)':^19}] Training Metrics: {tmp_top_metric} (top) | "
            f"{tmp_bottom_metric} (bottom) | {tmp_time_metric} (time)"
            )
    elif board == "PYNQ":
        tmp_power_metric, tmp_time_metric = online_models.get_metrics()
        print(
            f"[{'Training Thread (r)':^19}] Training Metrics: {tmp_power_metric} (power) | "
            f"{tmp_time_metric} (time)"
            )

    # Close shared memories (unlink if desired)
    buffers.clean(unlink=True)

    print(f"[{'Training Thread (r)':^19}] Thread terminated.")


def training_thread_ram_func_old(online_models, tcp_socket, board, cpu_usage):
    """Train models at run-time. (or test is the c code wants to check metrics)
    (online_data stored in ram-backed mmap'ed files)
    """

    # Wait for the client to connect via socket
    tcp_socket.wait_connection()
    print(f"[{'Training Thread (r)':^19}] TCP socket connected.")

    # Get number of training iterations
    notification = tcp_socket.recv_data()
    print(f"notification: {notification}")
    print(f"[{'Training Thread (r)':^19}]", datetime.now(timezone.utc))
    # Unpack the received binary data into an integer value
    number_iterations = struct.unpack("i", notification)[0]
    print(f"num_measurements: {number_iterations}")
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
    while notification != 0:

        notification = tcp_socket.recv_data()
        print(f"notification: {notification}")
        print(f"[{'Training Thread (r)':^19}]", datetime.now(timezone.utc))

        # Unpack the received binary data into an integer value
        received_data = struct.unpack("i", notification)[0]

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

        print(f"num_measurements: {num_measurements}")
        if num_measurements == 0:
            break
        # Get number of measurements to train with
        # num_measurements = int(notification)

        if i == 0:
            # Time measurement logic
            t_start = time.time()
        for _ in range(num_measurements):

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
                print(f"[{'Training Thread (r)':^19}] No useful obs")
                t_inter7 = time.time()
                t_interv += t_inter7-t_inter0
                num_useless_obs += 1

                # Debug
                t_copy_buf += t_inter1 - t_inter0
                t_gen_obs += t_inter2 - t_inter1
                t_toggle += t_inter3 - t_inter2
                continue

            # Create dataframe just for this online_data file
            dataframe = odp.generate_dataframe_from_observations(
                generated_obs,
                board,
                cpu_usage)

            t_inter4 = time.time()

            # DEBUG: Checking with observations are generated
            # print(curr_output_path)
            # print(traces_path)
            # print(curr_traces_file)
            # print(dataframe)

            t_inter5 = time.time()

            train_number_raw_observations = len(dataframe)
            # remove nan (happens due to not using mutex for cpu_usage)
            dataframe = dataframe.dropna()

            train_number_observations = len(dataframe)

            print(f"Train NaN rows: {train_number_raw_observations - train_number_observations}")

            if len(dataframe.index) < 1:
                num_useless_obs += 1
                continue

            # Different behaviour depending if is training or testing
            if is_training:
                # Learn batch with the online models
                online_models.train_s(dataframe, lock)
            else:
                # Test batch with the online models
                test_metrics = online_models.test_s(dataframe, lock, test_metrics)

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
                    f"[{'Training Thread (r)':^19}] Metrics: {tmp_top_metric} (top) | "
                    f"{tmp_bottom_metric} (bottom) | {tmp_time_metric} (time)"
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
                    f"[{'Training Thread (r)':^19}] Metrics: {tmp_power_metric} (power) | "
                    f"{tmp_time_metric} (time)"
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
        print(f"[{'Training Thread (r)':^19}] Number of trainings: {i}")

        print(
            f"[{'Training Thread (r)':^19}] Total Elapsed Time (s):    "
            f"(total) {t_end - t_start:016.9f} | (ratio) {(t_end - t_start) / i:012.9f}"
            )
        print(
            f"[{'Training Thread (r)':^19}] Interval Elapsed Time (s): "
            f"(total) {t_interv:016.9f} | (ratio) {t_interv / i:012.9f}"
            )

        # Debug
        print(
            f"[{'Training Thread (r)':^19}] t_copy_buf (s)           : "
            f"(total) {t_copy_buf:016.9f} | (ratio) {t_copy_buf / i:012.9f}"
            )
        print(
            f"[{'Training Thread (r)':^19}] t_gen_obs (s)            : "
            f"(total) {t_gen_obs:016.9f} | (ratio) {t_gen_obs / i:012.9f}"
            )
        print(
            f"[{'Training Thread (r)':^19}] t_toggle (s)             : "
            f"(total) {t_toggle:016.9f} | (ratio) {t_toggle / i:012.9f}"
            )
        print(
            f"[{'Training Thread (r)':^19}] t_gen_dataframe (s)      : "
            f"(total) {t_gen_dataframe:016.9f} | "
            f"(ratio) {t_gen_dataframe / (i - num_useless_obs):012.9f}"
            )
        print(
            f"[{'Training Thread (r)':^19}] t_print_df (s)           : "
            f"(total) {t_print_df:016.9f} | (ratio) {t_print_df / (i - num_useless_obs):012.9f}"
            )
        print(
            f"[{'Training Thread (r)':^19}] t_train (s)              : "
            f"(total) {t_train:016.9f} | (ratio) {t_train / (i - num_useless_obs):012.9f}"
            )
        print(
            f"[{'Training Thread (r)':^19}] t_metrics (s)            : "
            f"(total) {t_metrics:016.9f} | (ratio) {t_metrics / (i - num_useless_obs):012.9f}"
            )
    else:
        print(f"[{'Training Thread (r)':^19}] No processing was made")

    if board == "ZCU":
        tmp_top_metric, tmp_bottom_metric, tmp_time_metric = \
            online_models.get_metrics()
        print(
            f"[{'Training Thread (r)':^19}] Training Metrics: {tmp_top_metric} (top) | "
            f"{tmp_bottom_metric} (bottom) | {tmp_time_metric} (time)"
            )
    elif board == "PYNQ":
        tmp_power_metric, tmp_time_metric = online_models.get_metrics()
        print(
            f"[{'Training Thread (r)':^19}] Training Metrics: {tmp_power_metric} (power) | "
            f"{tmp_time_metric} (time)"
            )

    # Close shared memories (unlink if desired)
    buffers.clean(unlink=True)

    print(f"[{'Training Thread (r)':^19}] Thread terminated.")


def prediction_thread_func(online_models, tcp_socket, board, cpu_usage):
    """Use models to make a prediction at run-time."""

    print(f"[{'Prediction Thread':^19}] In")

    # Wait for the client to connect via socket
    tcp_socket.wait_connection()
    print(f"[{'Prediction Thread':^19}] TCP socket connected.")

    t_interv = 0
    buff = b'1'
    i = 0
    # Keep making predictions
    while buff != b'0':

        buff = tcp_socket.recv_data()
        print(f"buffer: {buff}")
        print(f"[{'Prediction Thread':^19}]", datetime.now(timezone.utc))

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
                    online_models.predict_one(features)

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
                    online_models.predict_one(features)

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
                    f"[{'Prediction Thread':^19}] Training Metrics: {tmp_top_metric} (top) | "
                    f"{tmp_bottom_metric} (bottom) | {tmp_time_metric} (time)"
                    )
            elif board == "PYNQ":
                tmp_power_metric, tmp_time_metric = online_models.get_metrics()
                print(
                    f"[{'Prediction Thread':^19}] Training Metrics: {tmp_power_metric} (power) | "
                    f"{tmp_time_metric} (time)"
                    )

    # Close the socket
    tcp_socket.close_connection()
    # Time measurement logic
    t_end = time.time()

    # Print useful information
    if i > 0:  # Take care of division by zero
        print(f"[{'Prediction Thread':^19}] Interval Elapsed Time (s):", t_interv, (t_interv)/i)
        print(
            f"[{'Prediction Thread':^19}] Total Elapsed Time (s):", t_end-t_start, (t_end-t_start)/i
            )
        print(f"[{'Prediction Thread':^19}] Number of predictions: {i}")

    else:
        print(f"[{'Prediction Thread':^19}] No predictions was made")

    print(f"[{'Prediction Thread':^19}] Thread terminated.")


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

    print(f"CPU usage: {cpu_usage_flag}")

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
                                           struct.calcsize("i"),
                                           struct.calcsize("i"))
    print(f"[{'Main Thread':^19}] TCP socket created.")

    # Create TCP server socket used to trigger the models' predictoin process
    tcp_prediction_socket = ipc.ServerSocketTCP("my_prediction_socket",
                                                ct.sizeof(Features),
                                                ct.sizeof(Prediction))
    print(f"[{'Main Thread':^19}] TCP socket created.")

    # Create thread safe lock
    lock = threading.Lock()

    # Initialize the online models
    incremental_models = om.OnlineModels(board=args.board, lock=lock)
    print(f"[{'Main Thread':^19}] Online Models have been successfully initialized.")

    # Create the training and prediction threads
    training_thread = threading.Thread(
        target=training_thread_func,
        args=(incremental_models,
              tcp_train_socket,
              args.board,
              cpu_usage_flag)
    )
    prediction_thread = threading.Thread(
        target=prediction_thread_func,
        args=(incremental_models,
              tcp_prediction_socket,
              args.board,
              cpu_usage_flag)
    )
    print(f"[{'Main Thread':^19}] Both threads have been successfully created.")

    # Start both threads
    training_thread.start()
    prediction_thread.start()
    print(f"[{'Main Thread':^19}] Both threads have been successfully started.")

    # Ensure all threads have finished
    training_thread.join()
    prediction_thread.join()
    print(f"[{'Main Thread':^19}] Both threads have successfully finished.")

    # Set observations file path
    # (there because is executed from ssh without permission in folder outputs)
    # Solvable by changing the ownership of that folder
    # observations_path = "/home/artico3/runtime_observations.pkl"

    # Save the dataframe in a file
    # df.to_pickle(observations_path)
