from border_ownership.square_part import Square_Part_Analyzer
import matplotlib.pyplot as plt
from border_ownership.rf_finder import out_of_range
import numpy as np
import os
import hickle as hkl
from mpi4py import MPI
from kitti_settings import *

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def get_small_rf_rank(module, data):

    # Select small rf
    eg_rf = data['bo_info'][module]['rf'].iloc[0]
    mask = out_of_range(eg_rf.shape[0], eg_rf.shape[1], outmost_distance)
    small_rf = data['bo_info'][module]['rf'].apply(lambda x: np.all(x[mask] == 0))
    data['bo_info'][module] = data['bo_info'][module][small_rf]

    small_rf_id = data['bo_info'][module]['neuron_id']
    data['res_square_part_info'][module] = data['res_square_part_info'][module]
    data['res_square_part_info'][module]['neuron_id'].isin(small_rf_id)

    neural_rank_list = data['bo_info'][module]['bo_only_rank'].dropna().unique()
    
    return neural_rank_list

def split_and_compute_res(module, data, neural_rank_list, mpi_size):
    if rank == 0:
        splits = np.array_split(neural_rank_list, size)
    else:
        splits = None

    split_data = comm.scatter(splits, root=0)
    local_res_all_neuron = []

    for i, neural_rank in enumerate(split_data):
        print('processing neural_rank: {} \t total neuron {}'.format(neural_rank, len(neural_rank_list)))
        spa = Square_Part_Analyzer(module, neural_rank, rank_method='bo_only')
        spa.load_data(data)
        res_temp = spa.get_res_change(mode='all_by_name', is_zscore=False, substract=False)
        local_res_all_neuron.append(res_temp)

    local_res_all_neuron = np.array(local_res_all_neuron)
    gathered_res_all_neuron = comm.gather(local_res_all_neuron, root=0)
    if rank == 0:
        return gathered_res_all_neuron

#################### Hyperparameters ####################
module_list = ['E0', 'E1', 'E2', 'E3', 'R0', 'R1', 'R2', 'R3']
# module_list = ['E2', 'E3']
outmost_distance = 15

center_info_path = os.path.join(DATA_DIR_HOME, 'center_neuron_info.hkl')
res_square_part_path = os.path.join(DATA_DIR_HOME, 'res_square_part_{}.hkl'.format(outmost_distance))

########### Population analysis ##########
data = hkl.load(center_info_path)

res_all_module = {}
for module in module_list:
    neural_rank_list = get_small_rf_rank(module, data)
    gathered_res_all_neuron = split_and_compute_res(module, data, neural_rank_list, mpi_size=size)

    if rank == 0:
        gathered_res_all_neuron = [arr for arr in gathered_res_all_neuron if arr.size > 0] # skip empty arr

        if len(gathered_res_all_neuron) == 0: # no neuron
            res_all_neuron = np.array([])
        else:
            res_all_neuron = np.concatenate(gathered_res_all_neuron)

        res_all_module[module] = res_all_neuron.copy()

if rank == 0:
    hkl.dump(res_all_module, res_square_part_path)

