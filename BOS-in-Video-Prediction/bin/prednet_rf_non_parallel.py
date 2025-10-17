# compute prednet neural heatmap
import os
import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl
import tensorflow as tf
from tensorflow.python.client import device_lib

from border_ownership.rf_finder import RF_Finder_Local_Sparse_Noise_Small_Memory_Center
from border_ownership.agent import Agent, Agent_RF_Wraper

from kitti_settings import *

json_file = 'prednet_kitti_model.json'
json_file = os.path.join(WEIGHTS_DIR, json_file)
weights_file = 'prednet_kitti_weights.hdf5'
weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/' + weights_file)

num_frames = 4
input_shape = (128, 160, 3)
meta_batch_size = 256
batch_size = None
# output_mode_list = ['E0', 'E1', 'E2', 'E3', 'R0', 'R1', 'R2', 'R3']
output_mode_list = ['R3']
query_neural_id = (0, 0, 0)

def single_thread_rf(output_mode, gpu=None):
    heatmap_dir = os.path.join(DATA_DIR_HOME, 'heatmap_{}'.format(output_mode))
    center_heatmap_dir = os.path.join(DATA_DIR_HOME, 'heatmap_center_{}'.format(output_mode))

    sub = Agent()
    sub.read_from_json(json_file, weights_file)
    sub_warp = Agent_RF_Wraper(sub, num_frames, output_mode, meta_batch_size=meta_batch_size, batch_size=batch_size)

    rff = RF_Finder_Local_Sparse_Noise_Small_Memory_Center(heatmap_dir, sub_warp, input_shape)

    rff.search_rf_all_neuron()
    rff.keep_center_heatmap(center_heatmap_dir)

for om in output_mode_list:
    single_thread_rf(om)
