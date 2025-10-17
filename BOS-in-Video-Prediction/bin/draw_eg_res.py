# draw example neural response
from border_ownership.center_neuron_analyzer import Center_RF_Neuron_Analyzer, save_center_neuron_only
import border_ownership.neuron_info_processor as nip
from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import os
import hickle as hkl
from kitti_settings import *
import matplotlib as mpl
font_size = 6
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['axes.labelsize'] = font_size
mpl.rcParams['ytick.labelsize'] = font_size
mpl.rcParams['xtick.labelsize'] = font_size
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['xtick.labelsize'] = font_size
mpl.rcParams['ytick.labelsize'] = font_size


# #################### Hyperparameters ####################
# module_list = ['R0', 'R1', 'R2', 'R3', 'E0', 'E1', 'E2', 'E3']
n_neuron = 6
module_list = ['E2']
neural_rank_list_top = np.arange(1, n_neuron + 1, 1)
neural_rank_list_bottom = np.arange(-0, - n_neuron, -1)
height = 1.5
rf_smooth_sigma = 1
center_info_path = os.path.join(DATA_DIR_HOME, 'center_neuron_info.hkl')

# #################### Main ####################
data = hkl.load(center_info_path)
bo_info, res_info, stim_info, unique_orientation = data['bo_info'], data['res_info'], data['stim_info'], data['unique_orientation']

nipor = nip.Neuron_Info_Processor()

n_neuron = len(neural_rank_list_top)
fig_rf_total, ax_rf_total = plt.subplots(len(module_list) * 2, n_neuron, figsize=(height * n_neuron, height * len(module_list) * 2))
fig_res_total, ax_res_total = plt.subplots(len(module_list) * 2, n_neuron, figsize=(height * n_neuron, height * len(module_list) * 2))
fig_ori_total, ax_ori_total = plt.subplots(len(module_list) * 2, n_neuron, figsize=(height * n_neuron, height * len(module_list) * 2), subplot_kw={'projection': 'polar'})

for idx_module, module in enumerate(module_list):
    nipor.load_data(bo_info, res_info, stim_info, module, unique_orientation)
    neuron_ori = nipor.boi_orim
    
    idx_module_top = idx_module * 2
    idx_module_bottom = idx_module * 2 + 1
    for i in range(n_neuron):
        top_neural_rank = neural_rank_list_top[i]
        bottom_neural_rank = neural_rank_list_bottom[i]

        ax_rf_total[idx_module_top, i], ax_res_total[idx_module_top, i], ax_ori_total[idx_module_top, i] = nip.draw_one_neuron(nipor, top_neural_rank, ax_rf_total[idx_module_top, i], ax_res_total[idx_module_top, i], ax_ori_total[idx_module_top, i])
        ax_rf_total[idx_module_bottom, i], ax_res_total[idx_module_bottom, i], ax_ori_total[idx_module_bottom, i] = nip.draw_one_neuron(nipor, bottom_neural_rank, ax_rf_total[idx_module_bottom, i], ax_res_total[idx_module_bottom, i], ax_ori_total[idx_module_bottom, i])
        ax_res_total[idx_module_top, i].set_title(f'Rank {top_neural_rank}', loc='right')
        ax_res_total[idx_module_bottom, i].set_title(f'Rank {bottom_neural_rank}', loc='right')
        if module == 'E0':
            ax_res_total[idx_module_top, i].set_ylim([0.99, 1.08])
            ax_res_total[idx_module_bottom, i].set_ylim([0.99, 1.08])

    ax_rf_total[idx_module_top, 0].set_title('Module ' + module + ' Top 6', loc='left')
    ax_rf_total[idx_module_bottom, 0].set_title('Module ' + module + ' Bottom 6', loc='left')
    ax_res_total[idx_module_top, 0].set_title('Module ' + module + ' Top 6', loc='left')
    ax_res_total[idx_module_bottom, 0].set_title('Module ' + module + ' Bottom 6', loc='left')
    ax_ori_total[idx_module_top, 0].set_title('Module ' + module + ' Top 6', loc='left')
    ax_ori_total[idx_module_bottom, 0].set_title('Module ' + module + ' Bottom 6', loc='left')

fig_rf_total.tight_layout()
fig_res_total.tight_layout()
fig_ori_total.tight_layout()

fig_rf_total.savefig(os.path.join(FIGURE_DIR, 'stacked_eg_neuron_rf.svg'), format='svg')
fig_res_total.savefig(os.path.join(FIGURE_DIR, 'stacked_eg_neuron_res.svg'), format='svg')
fig_ori_total.savefig(os.path.join(FIGURE_DIR, 'stacked_eg_neuron_ori.svg'), format='svg')
plt.show()

########## Show example four stimuli
# ori = 18
# four_square = stim_info[stim_info['orientation'] == ori]['image'].to_numpy()
# height, width = four_square[0].shape[:2]
# beta_gamma = stim_info[stim_info['orientation'] == ori][['beta', 'gamma']].to_numpy()
# four_color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
# fig, ax = plt.subplots(1, 4, figsize=(4, 1))
# for i in range(4):
#     ax[i].imshow(four_square[i])
#     ax[i].set_title(beta_gamma[i].astype(np.int))
#     ax[i].axis('off')
#     ax[i].add_patch(Rectangle((0, 0), width, height, fill=False, edgecolor=four_color[i], lw=6))
# plt.show()
