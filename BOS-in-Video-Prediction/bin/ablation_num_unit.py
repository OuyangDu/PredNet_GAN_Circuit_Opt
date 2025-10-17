import os
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from border_ownership.ploter import error_bar_plot, compute_y_and_ybound
import hickle as hkl
from border_ownership.ablation_visu import get_mse, plot_module, plot_modules
from kitti_settings import *

########################################
# merge different random boot
########################################
random_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # random_id for random initialization
data = {}
for rid in random_id:
    ########## Square dataset ##########
    # save_head = 'square'
    # data_path = os.path.join(RESULTS_SAVE_DIR, f"square_ablate_diff_num_units_random{rid}_10boot.hkl")
    ########################################

    ########## Natural video dataset ##########
    # save_head = 'natural'
    # data_path = os.path.join(RESULTS_SAVE_DIR, 'ablation_evaluate_backup', f"kitti_ablate_diff_num_units_random{rid}_10boot.hkl")
    ########################################

    ########## Translating natural video dataset ##########
    # save_head = 'trans'
    # data_path = os.path.join(RESULTS_SAVE_DIR, 'ablation_evaluate_backup', f"{save_head}_ablate_diff_num_units_random{rid}.hkl")
    ########################################

    ########## random square dataset ##########
    # save_head = 'random'
    # data_path = os.path.join(RESULTS_SAVE_DIR, 'ablation_evaluate_backup', f"{save_head}_ablate_diff_num_units_random{rid}.hkl")
    ########################################

    data_ = hkl.load(data_path)
    # remove no_ablation key, compute the relative prediction MSE
    no_ablation = data_['no_ablation']
    for key in data_.keys():
        if key != 'no_ablation':
            data_[key] = (np.array(data_[key]) - np.array(no_ablation)) / np.array(no_ablation) # relative prediction MSE
    del data_['no_ablation']
    # merge data
    for key in data_.keys():
        if key not in data:
            data[key] = []
        data[key].append(np.mean(data_[key])) # mean over 10 bootstraps
# convert to numpy array
for key in data.keys(): data[key] = np.array(data[key])

########################################
# plot
########################################
fig_save_path = os.path.join(RESULTS_SAVE_DIR, f'{save_head}_ablation_num_units.svg')
e_module = ['E0', 'E1', 'E2', 'E3']
r_module = ['R0', 'R1', 'R2', 'R3']

def plot_and_format_axes(ax, module, position, data):
    plot_module(data, module, fig, ax)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    if position[1] == 0 and position[0] == 0:
        ax.set_ylabel('Relative Prediction MSE')
        ax.legend(fontsize=12)
    ax.set_title(f'{module}', fontsize=14)

fig, ax = plt.subplots(2, 4, figsize=(10, 5))
for i, module in enumerate(e_module):
    plot_and_format_axes(ax[0, i], module, position=(0, i), data=data)

for i, module in enumerate(r_module):
    plot_and_format_axes(ax[1, i], module, position=(1, i), data=data)

fig.tight_layout()
fig.savefig(fig_save_path)
plt.show()

