#!/bin/bash

# Remote (host) Script for Setup Execution
#
# Author      : Juan Encinas <juan.encinas@upm.es>
# Date        : October 2023
# Description : This script orchestrates multiple sequential executions of the
#               setup on the remote board. Initially start a python script that
#               receives the setup traces via a socket from the remote board
#               and subsequently execute the local_setup_execution.sh script
#               on the remote board via ssh. It also logs the output of each
#               program as well at gather the logs generated in the remote
#               board using scp.
#
#               Must be placed in the host where the setup tracse are stored (PC).

# Remote server details
remote_user="artico3"
remote_server="hostname.local"
remote_password="artico3"

# Path to the local executable on the remote server
remote_executable="local_setup_execution.sh"

# Directory on the remote server where you want to run the executable
remote_working_directory="/home/juan/setup/MachSuite"
remote_outputs_directory="$remote_working_directory/setup_execution_outputs"

# Path to the Python script that receives traces via socket
receive_traces_script="receive_traces_over_socket_server_tcp.py"

# Create a folder to stores the outputs and traces as well as logs and past traces and outputs
outputs_dir="/media/juan/HDD/tmp/pruebas_modelos_online/test"
mkdir -p $outputs_dir

past_traces_dir="${outputs_dir}/traces_history"
mkdir -p $past_traces_dir

# Create a folder to store logs
logs_dir="${outputs_dir}/logs"
mkdir -p $logs_dir

# List of commands to execute
# Add more kernels and remove olders and then first and keep adding more and more till all added
commands=("generate_initial_conditions.py -n 20000 -t 10 --range-executions 5 --base-number-executions 128 -s 42 --range-kernels 0 4" "runtime_train_operating_modes.py ram ZCU" "./machsuite"
          "generate_initial_conditions.py -n 20000 -t 10 --range-executions 5 --base-number-executions 128 -s 42 --range-kernels 4 8" "runtime_train_operating_modes.py ram ZCU" "./machsuite"
          "generate_initial_conditions.py -n 20000 -t 10 --range-executions 5 --base-number-executions 128 -s 42 --range-kernels 8 11" "runtime_train_operating_modes.py ram ZCU" "./machsuite"
          "generate_initial_conditions.py -n 20000 -t 10 --range-executions 5 --base-number-executions 128 -s 42 --range-kernels 0 4" "runtime_train_operating_modes.py ram ZCU" "./machsuite"
          "generate_initial_conditions.py -n 20000 -t 10 --range-executions 5 --base-number-executions 128 -s 42 --range-kernels 0 8" "runtime_train_operating_modes.py ram ZCU" "./machsuite"
          "generate_initial_conditions.py -n 20000 -t 10 --range-executions 5 --base-number-executions 128 -s 42 --range-kernels 0 11" "runtime_train_operating_modes.py ram ZCU" "./machsuite")


commands_per_iteration=3

# Print general info
echo "-------------------------------------"
echo " Number of commands: ${#commands[@]}"
echo " Number of commands per iteration: $commands_per_iteration"
echo " Number of iterations: $((${#commands[@]}/${commands_per_iteration}))"
echo "-------------------------------------"

# Loop through the list of arguments
for ((i=0; i<$((${#commands[@]}/${commands_per_iteration})); i++)); do

    # Print iteration info
    echo -e "\n----------------\n"
    echo -e " Iteration #$i\n"
    echo "----------------"

    # Get python script command and machsuite executable command
    initial_condition_script="${commands[$(($commands_per_iteration*$i))]}"
    runtime_script="${commands[$(($commands_per_iteration*$i+1))]}"
    machsuite_executable="${commands[$(($commands_per_iteration*$i+2))]}"

    # Print commands info
    echo "--------------------------------------------------------------------"
    echo " '$initial_condition_script'"
    echo " runtime command:  '$runtime_script'"
    echo " setup executable: '$machsuite_executable'"
    echo "--------------------------------------------------------------------"

    # Create outputs and traces folders
    mkdir -p "${outputs_dir}/traces"
    mkdir -p "${outputs_dir}/outputs"

    # Run the Python script that receives traces
    receive_traces_output_file="${logs_dir}/receive_traces_$i.txt"
    python3 -u $receive_traces_script $outputs_dir > $receive_traces_output_file 2>&1 &
    receive_traces_script_pid=$!
    echo "Executing the python traces reception script..."

    # Wait a second for the python script to open the socket
    sleep 1

    # Launch the ZCU local_executable.sh via ssh
    ssh_command_output_file="${logs_dir}/ssh_command_$i.txt"
    sshpass -p "$remote_password" ssh ${remote_user}@${remote_server} << EOF > $ssh_command_output_file 2>&1 &
    cd ${remote_working_directory}
    echo "$remote_password" | sudo -S ./${remote_executable} "${i}" "${initial_condition_script}" "${runtime_script}" "${machsuite_executable}"
EOF
    echo "Running the local executable remotely from ${remote_working_directory} on ${remote_server}..."

    # Wait for the ssh command to finish
    echo "Waiting for the remote command to be executed..."
    wait

    # The script will continue once the ssh command has completed
    echo "The ssh command (remote machsuite exeution) has finished."

    # Move the traces and outputs folder to another folder with an iteration-identified name
    mkdir -p "${past_traces_dir}/iteration_$i"
    mv "${outputs_dir}/traces" "${past_traces_dir}/iteration_$i"
    mv "${outputs_dir}/outputs" "${past_traces_dir}/iteration_$i"

    # Copy logs (local and remote)
    mkdir -p "${logs_dir}/iteration_$i/local"
    # Local
    mv $receive_traces_output_file "${logs_dir}/iteration_$i/local"
    mv $ssh_command_output_file "${logs_dir}/iteration_$i/local"
    # Remote
    sshpass -p "$remote_password" scp -r ${remote_user}@${remote_server}:$remote_outputs_directory "${logs_dir}/iteration_$i/remote"
    echo "Remote and local logs have been copied from <${remote_user}@${remote_server}:$remote_outputs_directory> to <${logs_dir}/iteration_$i>"

    # Remove remote folders after copied
    sshpass -p "$remote_password" ssh ${remote_user}@${remote_server} "echo '$remote_password' | sudo -S rm -r '${remote_outputs_directory}'"
    echo -e "\nRemote log folder removed."

done

# Wait for all background jobs to finish before the script exits
wait
echo "The whole execution queue has finished!"
