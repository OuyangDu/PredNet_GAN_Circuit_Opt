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
from mpi4py import MPI

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # if you have multiple gpus, please comment this line and the same line in kitti_settings_lc.py

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type=='GPU']

gpus = get_available_gpus()
n_gpus = len(gpus)
print('number of gpus {}'.format(n_gpus))

json_file = 'prednet_kitti_model.json'
json_file = os.path.join(WEIGHTS_DIR, json_file)
weights_file = 'prednet_kitti_weights.hdf5'
weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/' + weights_file)

num_frames = 4
input_shape = (128, 160, 3)
meta_batch_size = None
batch_size = 128
output_mode_list = ['E0', 'E1', 'E2', 'E3', 'R0', 'R1', 'R2', 'R3']
query_neural_id = (0, 0, 0)

def single_thread_rf(output_mode, gpu):
    heatmap_dir = os.path.join(DATA_DIR_HOME, 'heatmap_{}'.format(output_mode))
    center_heatmap_dir = os.path.join(DATA_DIR_HOME, 'heatmap_center_{}'.format(output_mode))

    sub = Agent(gpu)
    sub.read_from_json(json_file, weights_file)
    sub_warp = Agent_RF_Wraper(sub, num_frames, output_mode, meta_batch_size=meta_batch_size, batch_size=batch_size)

    rff = RF_Finder_Local_Sparse_Noise_Small_Memory_Center(heatmap_dir, sub_warp, input_shape)

    rff.search_rf_all_neuron()
    rff.keep_center_heatmap(center_heatmap_dir)

for i in np.arange(0, len(output_mode_list), size): # parallel on cpu, but all threads run on the same gpu
    gpu = gpus[(i + rank)%n_gpus]
    single_thread_rf(output_mode_list[(i + rank)], gpu)

