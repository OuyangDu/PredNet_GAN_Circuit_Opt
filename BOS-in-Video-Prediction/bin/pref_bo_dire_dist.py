# preferred direction distribution
import matplotlib.pyplot as plt
import numpy as np
import os
import hickle as hkl
from kitti_settings import *

# #################### Hyperparameters ####################
module_list = ['R0', 'R1', 'R2', 'R3', 'E0', 'E1', 'E2', 'E3']
neural_rank = 10
center_info_path = os.path.join(DATA_DIR_HOME, 'center_neuron_info.hkl')

data = hkl.load(center_info_path)
bo_info, res_info, stim_info, unique_orientation = data['bo_info'], data['res_info'], data['stim_info'], data['unique_orientation']

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 5))

# First row for 'R0', 'R1', 'R2', 'R3'
for i, r_key in enumerate(['R0', 'R1', 'R2', 'R3']):
    bom = bo_info[r_key][bo_info[r_key]['is_bo']]
    total_units = bom['boi_pref_dire'].count()
    bom['boi_pref_dire'].hist(ax=axes[0, i], bins=20)
    axes[0, i].set_title(f'{r_key} (n = {total_units})')

    if i == 0:  # Leftmost axis for 'R' row
        axes[0, i].set_ylabel('Number of BO units')

# Second row for 'E0', 'E1', 'E2', 'E3'
for i, e_key in enumerate(['E0', 'E1', 'E2', 'E3']):
    bom = bo_info[e_key][bo_info[e_key]['is_bo']]
    total_units = bom['boi_pref_dire'].count()
    bom['boi_pref_dire'].hist(ax=axes[1, i], bins=20)
    axes[1, i].set_title(f'{e_key} (n = {total_units})')
    axes[1, i].set_xlabel('Preferred BO direction')

    if i == 0:  # Leftmost axis for 'E' row
        axes[1, i].set_ylabel('Number of BO units')

fig.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, 'pref_bo_dire_dist.svg'), bbox_inches='tight')
plt.show()
