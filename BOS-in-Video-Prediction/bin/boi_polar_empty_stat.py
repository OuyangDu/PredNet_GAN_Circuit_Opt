import border_ownership.neuron_info_processor as nip
from statsmodels.stats.proportion import proportion_confint
import border_ownership.ploter as ploter
from border_ownership.border_response_analysis import compute_BO, permute_beta_axis
from border_ownership.util import df_to_arr
from border_ownership.para_robust import get_res_para_ori
import matplotlib.pyplot as plt
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
import hickle as hkl
from kitti_settings import *

import cProfile, pstats

def compute_boi(arr):
    arr = arr.swapaxes(-2, -1) # (alpha, beta, gamma, time, size)
    boi = compute_BO(arr) # (alpha, time, size)
    boi = np.mean(boi, axis=(1, 2)) # (alpha)
    return boi

def permutation_test(boi, arr, n_shuffle=5000):
    boi_shuffle = np.empty((n_shuffle, len(boi)), dtype=np.float32)

    for i in range(n_shuffle):
        arr_shuffle = permute_beta_axis(arr, for_each_axis=[0, 2, 4])
        boi_shuffle[i] = compute_boi(arr_shuffle)

    # two-tailed test for each orientation
    p_value = np.empty(len(boi), dtype=np.float32)
    for i in range(len(boi)):
        more_extreme = np.sum(boi[i] < boi_shuffle[:, i]) / n_shuffle
        p_value[i] = 2 * min(more_extreme, 1 - more_extreme) # (alpha)
    return p_value

def convert_dire(boi, orientation):
    boi_bool = boi > 0

    dire = []
    for i, ori in enumerate(orientation):
        if boi_bool[i]:
            dire.append(ori)
        else:
            dire.append(ori + 180)
    return np.array(dire)

#################### Hyperparameters ####################
module = 'E2'
neural_rank = 10
center_info_path = os.path.join(DATA_DIR_HOME, 'center_neuron_info.hkl')

# #################### Main ####################
data = hkl.load(center_info_path)
bo_info, res_info, stim_info, unique_orientation = data['bo_info'], data['res_info'], data['stim_info'], data['unique_orientation']
res_size_info, stim_size_info = data['res_size_info'], data['stim_size_info'] # used for testing statistical significance of BOI along a orientation

nipor = nip.Neuron_Info_Processor()
nipor.load_data(bo_info, res_info, stim_info, module, unique_orientation)
_, neuron_res_info = nipor.get_target_neuron_info(neural_rank, rank_method='bo_only')

arr, unique_values = df_to_arr(neuron_res_info, value_name='response', condition_names=['orientation', 'beta', 'gamma', 'size'])
boi = compute_boi(arr)
p_value = permutation_test(boi, arr)
dire = convert_dire(boi, unique_values[0]) # (alpha)

significant_boi = boi[p_value < 0.05]
significant_dire = dire[p_value < 0.05]

non_significant_boi = boi[p_value >= 0.05]
non_significant_dire = dire[p_value >= 0.05]

fig, ax = plt.subplots(figsize=(3, 3), subplot_kw={'projection': 'polar'})
ax = ploter.plot_polar_boi(significant_boi, significant_dire, ax=ax)
ax = ploter.plot_polar_boi(non_significant_boi, non_significant_dire, ax=ax, fillstyle='none')
ax.tick_params(axis='y', labelsize=22)
fig.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, 'boi_polar_empty_stat.svg'))
plt.show()
