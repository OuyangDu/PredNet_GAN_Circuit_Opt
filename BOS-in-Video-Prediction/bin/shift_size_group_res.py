import border_ownership.neuron_info_processor as nip
import border_ownership.ploter as ploter
import matplotlib.pyplot as plt
from border_ownership.para_robust import get_res_para
import numpy as np
import os
import hickle as hkl
from kitti_settings import *

def compute_boi_p_value(boi, n_boot=10000):
    '''
    boi: (n_neuron, n_para) numpy array
    '''
    boi_dis = []
    for _ in range(n_boot):
        idx = np.random.choice(boi.shape[0], boi.shape[0], replace=True)
        boi_boot = boi[idx]
        boi_boot_mean = np.mean(boi_boot)
        boi_dis.append(boi_boot_mean)
    zero_quantile = np.sum(np.array(boi_dis) <= 0) / len(boi_dis)
    p_value = zero_quantile
    return p_value


def get_group_data(nipor, mode='shift'):
    n_bo = nipor.bo_infom['bo_only_rank'].max() # get the maximum number of BO units
    mean_response_beta_pref_list, mean_response_beta_npref_list, para_beta_list = [], [], [] # data collectors
    for nr in range(1, int(n_bo)+1):
        neuron_bo_info, neuron_res_info = nipor.get_target_neuron_info(nr, rank_method='bo_only')
        mean_response_beta_pref, mean_response_beta_npref, para_beta = get_res_para(neuron_bo_info, neuron_res_info, mode=mode)
        mean_response_beta_pref_list.append(mean_response_beta_pref)
        mean_response_beta_npref_list.append(mean_response_beta_npref)
        para_beta_list.append(para_beta)
    norm_p = np.mean(mean_response_beta_pref_list, axis=1)
    norm_np = np.mean(mean_response_beta_npref_list, axis=1)
    norm = (norm_p + norm_np) / 2

    mrb_pref_norm =  mean_response_beta_pref_list / norm[:, None]; mrb_npref_norm = mean_response_beta_npref_list / norm[:, None]
    # mrb_pref_norm = np.array(mean_response_beta_pref_list);
    # mrb_npref_norm = np.array(mean_response_beta_npref_list)
    return mrb_pref_norm.astype(float), mrb_npref_norm.astype(float), para_beta_list[0].astype(float)

def compute_boi(mrb_pref, mrb_npref):
    boi = (mrb_pref - mrb_npref) / (mrb_pref + mrb_npref) * 2
    return boi

# #################### Hyperparameters ####################
module_list = ['E0', 'E1', 'E2', 'E3', 'R0', 'R1', 'R2', 'R3']
# module_list = ['E2']
remove_outlier = False
error_mode = 'quantile'; mean_mode = 'median'
center_info_path = os.path.join(DATA_DIR_HOME, 'center_neuron_info.hkl')
data = hkl.load(center_info_path)
bo_info, res_info, stim_info, unique_orientation = data['bo_info'], data['res_info'], data['stim_info'], data['unique_orientation']
res_shift_info, stim_shift_info = data['res_shift_info'], data['stim_shift_info']
res_size_info, stim_size_info = data['res_size_info'], data['stim_size_info']

#################### Different shift/size ####################
# nipor = nip.Neuron_Info_Processor()
# ### get data
# shift_size_data = {}
# for module in module_list:
#     nipor.load_data(bo_info, res_shift_info, stim_shift_info, module, unique_orientation)
#     mrb_pref_shift, mrb_npref_shift, para_shift = get_group_data(nipor, mode='shift')
#     nipor.load_data(bo_info, res_size_info, stim_size_info, module, unique_orientation)
#     mrb_pref_size, mrb_npref_size, para_size = get_group_data(nipor, mode='size')
#     shift_size_data[module] = {'mrb_pref_shift': mrb_pref_shift, 'mrb_npref_shift': mrb_npref_shift, 'para_shift': para_shift, 'mrb_pref_size': mrb_pref_size, 'mrb_npref_size': mrb_npref_size, 'para_size': para_size}

# hkl.dump(shift_size_data, os.path.join(DATA_DIR_HOME, 'shift_size_group_res.hkl'))

### plot
shift_size_data = hkl.load(os.path.join(DATA_DIR_HOME, 'shift_size_group_res.hkl'))

# response figure
fig, axes = plt.subplots(2, 4, figsize=(12, 5))
ax = axes.flatten()
for i, module in enumerate(module_list):
    data = shift_size_data[module]
    ploter.error_bar_plot(data['para_shift'], data['mrb_pref_shift'].T, fig=fig, ax=ax[i], color='tab:blue', error_mode=error_mode, mean_mode=mean_mode, remove_outlier=remove_outlier)
    ploter.error_bar_plot(data['para_shift'], data['mrb_npref_shift'].T, fig=fig, ax=ax[i], color='tab:green', error_mode=error_mode, mean_mode=mean_mode, remove_outlier=remove_outlier)
    n_unit = data['mrb_pref_shift'].shape[0]
    ax[i].set_title(f'{module}: {n_unit}', fontsize=16)
    if i in [0, 4]:
        ax[i].set_ylabel('Unit activation')
    if i in [4, 5, 6, 7]:
        ax[i].set_xlabel('Square position (pixel)')
    ax[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax[0].legend(['Preferred', 'Non-preferred'], loc='upper left', fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, 'shift_group_res.svg'))

fig, axes = plt.subplots(2, 4, figsize=(12, 5))
ax = axes.flatten()
for i, module in enumerate(module_list):
    data = shift_size_data[module]
    ploter.error_bar_plot(data['para_size'], data['mrb_pref_size'].T, fig=fig, ax=ax[i], color='tab:blue', error_mode=error_mode, mean_mode=mean_mode, remove_outlier=remove_outlier)
    ploter.error_bar_plot(data['para_size'], data['mrb_npref_size'].T, fig=fig, ax=ax[i], color='tab:green', error_mode=error_mode, mean_mode=mean_mode, remove_outlier=remove_outlier)
    n_unit = data['mrb_pref_size'].shape[0]
    ax[i].set_title(f'{module}: {n_unit}', fontsize=16)
    if i in [0, 4]:
        ax[i].set_ylabel('Unit activation')
    if i in [4, 5, 6, 7]:
        ax[i].set_xlabel('Square size (pixel)')
    ax[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
fig.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, 'size_group_res.svg'))

# BOI figure
fig, axes = plt.subplots(2, 4, figsize=(12, 5))
ax = axes.flatten()
for i, module in enumerate(module_list):
    data = shift_size_data[module]
    boi = compute_boi(data['mrb_pref_shift'], data['mrb_npref_shift'])
    p_value = compute_boi_p_value(boi)
    print(f'{module}: {p_value}')
    ploter.error_bar_plot(data['para_shift'], boi.T, fig=fig, ax=ax[i], color='k', error_mode=error_mode, mean_mode=mean_mode, remove_outlier=remove_outlier)
    xmin, xmax = ax[i].get_xlim()
    ax[i].hlines(0, xmin=xmin, xmax=xmax, color='k', linestyle='--')
    n_unit = data['mrb_pref_shift'].shape[0]
    ax[i].set_title(f'{module}: {n_unit}', fontsize=16)
    if i in [0, 4]:
        ax[i].set_ylabel('BOI')
    if i in [4, 5, 6, 7]:
        ax[i].set_xlabel('Square position (pixel)')
    ax[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
fig.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, 'boi_shift_group_res.svg'))

fig, axes = plt.subplots(2, 4, figsize=(12, 5))
ax = axes.flatten()
for i, module in enumerate(module_list):
    data = shift_size_data[module]
    boi = compute_boi(data['mrb_pref_size'], data['mrb_npref_size'])
    p_value = compute_boi_p_value(boi)
    print(f'{module}: {p_value}')
    ploter.error_bar_plot(data['para_size'], boi.T, fig=fig, ax=ax[i], color='k',  error_mode=error_mode, mean_mode=mean_mode, remove_outlier=remove_outlier)
    xmin, xmax = ax[i].get_xlim()
    ax[i].hlines(0, xmin=xmin, xmax=xmax, color='k', linestyle='--')
    n_unit = data['mrb_pref_size'].shape[0]
    ax[i].set_title(f'{module}: {n_unit}', fontsize=16)
    if i in [0, 4]:
        ax[i].set_ylabel('BOI')
    if i in [4, 5, 6, 7]:
        ax[i].set_xlabel('Square size (pixel)')
    ax[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
fig.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, 'boi_size_group_res.svg'))

plt.show()
