import os
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['legend.frameon'] = False  # Removes the legend's background box
mpl.rcParams['legend.fontsize'] = 'small'  # Sets the legend font size to small

PYTHON_COMPUTING_DEVICE = os.environ['PYTHON_COMPUTING_DEVICE']

if PYTHON_COMPUTING_DEVICE == 'local_computer':
    from kitti_settings_lc import *
elif PYTHON_COMPUTING_DEVICE == 'high_performance_computer':
    from kitti_settings_hpc import *
