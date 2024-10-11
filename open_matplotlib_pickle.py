#!/usr/bin/env python3

"""
Open Matplotlib figures saved in a pickle file

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : October 2023
Description : This script opens and displays a matplotlib figure stored in a
              pickle file located on a path defined by the user.
"""
import argparse
import sys
import pickle
import matplotlib.pyplot as plt

# Parse arguments
parser = argparse.ArgumentParser()

# This is the correct way to handle accepting multiple arguments.
# '+' == 1 or more.
# '*' == 0 or more.
# '?' == 0 or 1.
# An int is an explicit number of arguments to accept.
# Indicate the path of the file containing the datasets.
parser.add_argument('-i', dest="figure_path", help='<Required> Path to the Matplotlib figure to open', required=True)

args = parser.parse_args(sys.argv[1:])

# Load the figure using pickle
with open(args.figure_path, 'rb') as f:
    loaded_fig = pickle.load(f)

"""

# Create a new Matplotlib FigureCanvas and attach it to a GUI backend
new_fig = plt.figure()
new_fig.canvas = loaded_fig.canvas """

# Show or interact with the reloaded figure
plt.show()
