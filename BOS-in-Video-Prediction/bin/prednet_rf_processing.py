# non-parallel version
# # use the prednet response to sparse noise (generated from prednet_rf.py) to compute the RF
# from border_ownership.rf_finder import RF_Finder_Local_Sparse_Noise_Small_Memory
# import numpy as np
# import hickle as hkl
# import matplotlib.pyplot as plt
# from kitti_settings import *

# #################### Hyperparameters ####################
# output_mode = ['E1', 'R0', 'R1', 'R2', 'R3', 'E0', 'E2', 'E3']
# # output_mode = ['R0']
# outmost_distance = 40 // 2

# center_neuron_dict = {}
# for om in output_mode:
#     print('working on output mode {}'.format(om))
#     heatmap_dir = os.path.join(DATA_DIR_HOME, 'heatmap_{}'.format(om))
#     rfer = RF_Finder_Local_Sparse_Noise_Small_Memory(data_dir=heatmap_dir)
#     # rfer.load_heatmap_all()
#     neuron_id_list, heatmap_list, rf_list = rfer.obtain_central_RF_neuron(outmost_distance=outmost_distance)
#     center_neuron_dict[om] = {'neuron_id_list': neuron_id_list, 'heatmap_list': heatmap_list, 'rf_list': rf_list}

# hkl.dump(center_neuron_dict, os.path.join(DATA_DIR_HOME, 'center_neuron_dict.hkl'))


# data = hkl.load(os.path.join(DATA_DIR_HOME, 'center_neuron_dict.hkl'))
# print(len(data['R0']['neuron_id_list']))
# print(data.keys())
# print(data['R0'].keys())
# for i in range(10):
#     plt.figure()
#     plt.imshow(data['R0']['rf_list'][i])
#     plt.show()

# Parallel version
from border_ownership.rf_finder import RF_Finder_Local_Sparse_Noise_Small_Memory
import numpy as np
import hickle as hkl
import matplotlib.pyplot as plt
from kitti_settings import *
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

outmost_distance = 20 // 2 # maximum radius of the RF
output_mode = ['R0', 'R1', 'R2', 'R3', 'E0', 'E1', 'E2', 'E3']
portion = len(output_mode) // size
# Split output_mode for each process
local_output_mode = output_mode[rank*portion:rank*portion+portion]

center_neuron_dict = {}
for om in local_output_mode:
    print(f'Process {rank} working on output mode {om}')
    heatmap_dir = os.path.join(DATA_DIR_HOME, f'heatmap_{om}')
    rfer = RF_Finder_Local_Sparse_Noise_Small_Memory(data_dir=heatmap_dir)
    neuron_id_list, heatmap_list, rf_list = rfer.obtain_central_RF_neuron(outmost_distance=outmost_distance)
    center_neuron_dict[om] = {
        'neuron_id_list': neuron_id_list, 
        'heatmap_list': heatmap_list, 
        'rf_list': rf_list
    }

# Gather the results from each process
all_dicts = comm.gather(center_neuron_dict, root=0)
if rank == 0:
    total_center_neuron_dict = {k: v for d in all_dicts for k, v in d.items()}
    hkl.dump(total_center_neuron_dict, os.path.join(DATA_DIR_HOME, 'center_neuron_dict.hkl'))
