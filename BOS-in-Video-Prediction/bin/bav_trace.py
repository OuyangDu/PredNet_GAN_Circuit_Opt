# draw example neural response
from border_ownership.center_neuron_analyzer import Center_RF_Neuron_Analyzer
import hickle as hkl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os
from kitti_settings import *

def get_bavt(cn_analyzer, module, t_step):
    bav = []
    for t in t_step:
        bav_t, _ = cn_analyzer.compute_bav(bav_mean_time_init=t, bav_mean_time_final=t+2)
        bav.append(bav_t[module])
    return bav
# #################### Hyperparameters ####################
center_neuron_rf_path = os.path.join(DATA_DIR_HOME, 'center_neuron_dict.hkl')
center_res_ori_path = os.path.join(DATA_DIR_HOME, 'square_bo_center_res_ori.npz')
center_res_shift_path = os.path.join(DATA_DIR_HOME, 'square_bo_center_res_shift.npz')
center_res_size_path = os.path.join(DATA_DIR_HOME, 'square_bo_center_res_size.npz')
center_res_square_part_path = os.path.join(DATA_DIR_HOME, 'square_bo_center_res_square_part.npz')

bavt_dict_path = os.path.join(DATA_DIR_HOME, 'bavt_dict.hkl')
figpath = os.path.join(FIGURE_DIR, 'bavt.svg')

tmax=20
module_list = ['R0', 'R1', 'R2', 'R3', 'E0', 'E1', 'E2', 'E3']
# #################### Main ####################
# cn_analyzer = Center_RF_Neuron_Analyzer()
# cn_analyzer.load_data(center_res_ori_path, center_neuron_rf_path)
# cn_analyzer.bav_permutation_test(n_permutation=1000) # compute is bo
# is_bo = cn_analyzer.is_bo

# t_step = np.arange(0, tmax, 1)

# bavt_dict = {}
# for module in module_list:
#     bavt = get_bavt(cn_analyzer, module, t_step)
#     bavt_dict[module] = np.array(bavt).T
#     bavt_dict[module] = bavt_dict[module][is_bo[module]] # bo unit only

# data = {'bavt_dict': bavt_dict, 't_step': t_step}
# hkl.dump(data, bavt_dict_path)
# exit()

data = hkl.load(bavt_dict_path)
bavt_dict = data['bavt_dict']; t_step = data['t_step']

fig, axes = plt.subplots(2, 4, figsize=(10, 5))
axes = axes.flatten()
for i, module in enumerate(module_list):
    bav = bavt_dict[module]

    mean = np.mean(bav, axis=0)
    sem = np.std(bav, axis=0) / np.sqrt(bav.shape[0])

    # plotting the time traces with error bands
    ax = axes[i]
    ax.plot(t_step, mean, color='k')
    ax.fill_between(t_step, mean-sem, mean+sem, color='k', alpha=0.2)
    ax.set_title(module, fontsize=15)

# set y label
for ax in axes.flatten():
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))
    ax.yaxis.set_major_formatter(formatter)

    ax.yaxis.get_offset_text().set_fontsize(13)

# add x and y labels
x_label_id = [4, 5, 6, 7]
y_label_id = [0, 4]
for xi in x_label_id:
    axes[xi].set_xlabel('Time', fontsize=13)
for yi in y_label_id:
    axes[yi].set_ylabel('Bav', fontsize=13)
fig.tight_layout()
fig.savefig(figpath, format='svg')

plt.show()

