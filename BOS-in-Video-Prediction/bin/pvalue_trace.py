# draw example neural response
from border_ownership.center_neuron_analyzer import Center_RF_Neuron_Analyzer
import hickle as hkl
import numpy as np
import copy
from mpi4py import MPI
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from kitti_settings import *

#################### Hyperparameters ####################
center_neuron_rf_path = os.path.join(DATA_DIR_HOME, 'center_neuron_dict.hkl')
center_res_ori_path = os.path.join(DATA_DIR_HOME, 'square_bo_center_res_ori.npz')
center_res_shift_path = os.path.join(DATA_DIR_HOME, 'square_bo_center_res_shift.npz')
center_res_size_path = os.path.join(DATA_DIR_HOME, 'square_bo_center_res_size.npz')
center_res_square_part_path = os.path.join(DATA_DIR_HOME, 'square_bo_center_res_square_part.npz')
tmax=20

data_path = os.path.join(DATA_DIR_HOME, 'pvalue_t.hkl')
module_list = ['R0', 'R1', 'R2', 'R3', 'E0', 'E1', 'E2', 'E3']
# #################### Main ####################
# # MPI initialization
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

# cn_analyzer = Center_RF_Neuron_Analyzer()
# cn_analyzer.load_data(center_res_ori_path, center_neuron_rf_path)
# # Initial computation might only be needed once.
# cn_analyzer.bav_permutation_test()  # compute is below
# is_bo = copy.deepcopy(cn_analyzer.is_bo)

# t_step = np.arange(0, tmax, 1)
# t_step_split = np.array_split(t_step, size)  # Split the array amongst processes

# # Each process runs a chunk of the original loop
# pvalue_dict_local = defaultdict(list)
# for t in t_step_split[rank]:
#     cn_analyzer.bav_permutation_test(bav_mean_time_init=t, bav_mean_time_final=t+1)
#     for module in module_list:
#         pvalue_dict_local[module].append(cn_analyzer.bav_pvalue[module])

# # Gather results at root process
# pvalue_dict_gathered = comm.gather(pvalue_dict_local, root=0)

# # Only the root process should compile the results and save them
# if rank == 0:
#     pvalue_dict = defaultdict(list)
#     for pd in pvalue_dict_gathered:
#         for module in module_list:
#             pvalue_dict[module].extend(pd[module])

#     # Convert lists to numpy arrays and transpose
#     for module in module_list:
#         pvalue_dict[module] = np.array(pvalue_dict[module]).T[is_bo[module]]

#     data = {'pvalue': pvalue_dict, 't_step': t_step}
#     hkl.dump(data, data_path)

# # Finalize the MPI environment
# MPI.Finalize()
# exit()

data = hkl.load(data_path)
pvalue_dict = data['pvalue']; t_step = data['t_step']

fig, axes = plt.subplots(2, 4, figsize=(10, 5))
axes = axes.flatten()
for i, module in enumerate(module_list):
    pvalue = pvalue_dict[module]
    n_neuron = pvalue.shape[0]

    mean = np.mean(pvalue, axis=0)
    sem = np.std(pvalue, axis=0) / np.sqrt(pvalue.shape[0])

    # plotting the time traces with error bands
    ax = axes[i]
    ax.plot(t_step, mean, color='k')
    ax.scatter(t_step, mean, color='k', s=10)
    ax.fill_between(t_step, mean-sem, mean+sem, color='k', alpha=0.2)
    ax.set_title(module, fontsize=15)

# add x and y labels
x_label_id = [4, 5, 6, 7]
y_label_id = [0, 4]
for xi in x_label_id:
    axes[xi].set_xlabel('Time', fontsize=13)
for yi in y_label_id:
    axes[yi].set_ylabel('Pvalue of the Bav', fontsize=13)

fig.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, 'pvalue_t.svg'))
plt.show()
