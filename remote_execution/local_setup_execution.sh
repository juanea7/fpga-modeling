#!/bin/bash

# Local (ZCU) Script for Setup Execution
#
# Author      : Juan Encinas <juan.encinas@upm.es>
# Date        : October 2023
# Description : This script orchestrates all the steps for the proper execution
#               of the setup through the orderly execution of multiple programs
#               (generation of initial conditions, management of the online
#               models and actual execution of the setup). It also logs the
#               output of each of the programs.
#
#               Must be placed in the remote board where the setup is actually
#               executed (ZCU).

# Function to handle Ctrl+C
function handle_ctrl_c {
    echo "Ctrl+C pressed. Stopping background processes..."
    
    # Send Ctrl+C to the background Python initial condition generation script (if it's running)
    if [ -n "$initial_condition_script_pid" ]; then
        if ps -p "$initial_condition_script_pid" > /dev/null; then
            kill -TERM "$initial_condition_script_pid"
        fi
    fi

    # Send Ctrl+C to the background Python runtime script (if it's running)
    if [ -n "$runtime_script_pid" ]; then
        if ps -p "$runtime_script_pid" > /dev/null; then
            kill -TERM "$runtime_script_pid"
        fi
    fi

    # Send Ctrl+C to the background machsuite executable (if it's running)
    if [ -n "$machsuite_script_pid" ]; then
        if ps -p "$machsuite_script_pid" > /dev/null; then
            kill -TERM "$machsuite_script_pid"
        fi
    fi

    # Wait for background processes to exit
    wait

    echo "Background processes have been stopped."

    # The temporal shm files are removed, even though they should be removed by the python script
    rm /dev/shm/*
    exit 1
}

# Trap Ctrl+C and call the custom function
trap handle_ctrl_c INT

# Get the arguments (first is the iteration, rest are the commands)
args=("$@")

# Get the number of the actual iteration
iteration="${args[0]}"

# Get python script command and machsuite executable command
initial_condition_script="${args[1]}"
runtime_script="${args[2]}"
machsuite_executable="${args[3]}"

# Print iteration info
echo -e "\n----------------\n"
echo -e " Iteration #$iteration\n"
echo "----------------"

# Print commands info
echo "--------------------------------------------------------------------"
echo " '$initial_condition_script'"
echo " runtime command:   '$runtime_script'"
echo " setup executable: '$machsuite_executable'"
echo "--------------------------------------------------------------------"

# Enter the virtual environment
source env/bin/activate

# Create a folder to stores the output of each executable
outputs_dir="setup_execution_outputs"
mkdir -p $outputs_dir

# Launch initial condition generation
initial_condition_output_file="${outputs_dir}/initial_conditions_${iteration}.txt"
python -u $initial_condition_script > $initial_condition_output_file 2>&1 &
initial_condition_script_pid=$!
echo "Executing the python initial conditions generation script..."

# Launch the python script in the background that trains the models
run_time_output_file="${outputs_dir}/run_time_${iteration}.txt"
python -u $runtime_script > $run_time_output_file 2>&1 &
runtime_script_pid=$!
echo "Executing the python run time script..."

# Wait 10 seconds for the python interpreter to run
echo "Waiting 10s for the python interpreter to start-up..."
sleep 10

# Launch the setup
machsuite_output_file="${outputs_dir}/machsuite_${iteration}.txt"
unbuffer $machsuite_executable > $machsuite_output_file 2>&1 &
machsuite_script_pid=$!
echo "Executing the setup..."

# Wait for the python script and the machsuite executable running in the background
echo "Waiting for both the python script and the setup executable to finish..."
wait

# The script will continue once both have completed
echo "Both Python script and executable have finished."

# The temporal shm files are removed, even though they should be removed by the python script
rm /dev/shm/*
