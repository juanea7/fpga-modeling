# Incremental Learning Module

The `incremental_learning` module provides tools and classes for implementing and managing online learning models. It includes functionalities for processing data, managing shared memory buffers, and training/testing models in real-time.

## Incremental Learning Structure

### Submodules

- **execution_modes_buffers**
    - Handles different execution modes for training and testing models using shared memory buffers.

- **inter_process_communication**
    - Manages communication between processes, including socket communication for sending and receiving data.

- **online_data_processing**
    - Contains functions for processing online data, including generating observations and features from raw data.

- **online_models**
    - Implements various online learning models and their training/testing routines.

- **ping_pong_buffers**
    - Manages ping-pong buffers for sharing data among independent processes using shared memory.

## Example Scripts

There are several scripts available to demonstrate different ways to use the incremental learning module:

- **all_models_train.py**
    - This script trains all available online models using a predefined dataset and evaluates their performance.

- **online_traces_processing.py**
    - This script processes power and performance traces read from local files and generates a pickle file containing a dataframe with all processed observations.

- **open_matplotlib_processing.py**
    - This script opens and processes data using Matplotlib for visualization, allowing for real-time plotting of model performance.

- **postprocessing_model_data.py**
    - This script processes the online models' output data and generates error metrics figures. It includes functions for visualizing model data with and without time information.

- **runtime_adaptive_models.py**
    - This script adapts the models in real-time based on incoming data, adjusting parameters to improve performance dynamically. Needs to be run on the board (remember to copy the incremental_learning module.)

## Dependencies

For run-time usage on the board, this module requires the [FPGA Workload Manager](https://github.com/juanea7/fpga-workload-manager.git). This manager leverages the incremental learning module for workload modeling and management.

## Paper

This project is described in detail in the accompanying paper. You can read the paper [here](link_to_paper).