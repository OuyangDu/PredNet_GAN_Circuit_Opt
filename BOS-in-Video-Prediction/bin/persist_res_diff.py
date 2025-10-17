# please run quick_export_neuron_info_switch first
import border_ownership.neuron_info_processor as nip
import border_ownership.ploter as ploter
from collections import defaultdict
from border_ownership.persistent import Res_Diff_Calculator
import matplotlib.ticker as mticker
import border_ownership.persistent as pers
import matplotlib.pyplot as plt
import numpy as np
import os
import hickle as hkl
from kitti_settings import *

def plot_traces_module(data, module, fig=None, ax=None, error_mode='se', mean_mode='mean', remove_outlier=False):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.5))

    res_diff_dict, res_diff_s_dict, res_diff_ssf_dict = data['res_diff'], data['res_diff_s'], data['res_diff_ssf']
    time, switch_time = data['time'], data['switch_time'] # get time information
    # plot

    fig, ax = ploter.error_bar_plot(time, res_diff_dict[module].T, fig=fig, ax=ax, color='k', label='Square', error_band=True, remove_outlier=remove_outlier, error_mode=error_mode, line_style='--', mean_mode=mean_mode)
    fig, ax = ploter.error_bar_plot(time, res_diff_s_dict[module].T, fig=fig, ax=ax, color='tab:red', label='Square-Ambiguous', error_band=True, remove_outlier=remove_outlier, error_mode=error_mode, mean_mode=mean_mode)
    fig, ax = ploter.error_bar_plot(time, res_diff_ssf_dict[module].T, fig=fig, ax=ax, color='tab:blue', label='Square-Flip', error_band=True, remove_outlier=remove_outlier, error_mode=error_mode, mean_mode=mean_mode)
    
    ax.axvline(x=switch_time, color='black', linestyle='--')
    space = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.text(switch_time, ax.get_ylim()[0] - space, str(switch_time), va='top', ha='center', fontsize=15)
    ax.axhline(y=0, color='black', linestyle='--')
    n_unit = res_diff_dict[module].shape[0]
    ax.set_title(f'{module}: {n_unit}')

    return fig, ax

# #################### Hyperparameters ####################
module_list = ['R0', 'R1', 'R2', 'R3', 'E0', 'E1', 'E2', 'E3']
center_info_path = os.path.join(DATA_DIR_HOME, 'center_neuron_info.hkl')
error_mode = 'se'
mean_mode = 'mean'
remove_outlier = False

#################### Main ####################
data = hkl.load(center_info_path)
bo_info, unique_orientation = data['bo_info'], data['unique_orientation']

rdc = Res_Diff_Calculator(bo_info, data['res_info'], data['stim_info'], unique_orientation)
rdc_s = Res_Diff_Calculator(bo_info, data['res_switch_info'], data['stim_switch_info'], unique_orientation)
rdc_ssf = Res_Diff_Calculator(bo_info, data['res_switch_square_flip_info'], data['stim_switch_square_flip_info'], unique_orientation)

# res_diff_dict, res_diff_s_dict, res_diff_ssf_dict = defaultdict(list), defaultdict(list), defaultdict(list)
# for module in module_list:
#     max_rank = int(bo_info[module]['bo_only_rank'].max())
#     print('Module {}; max_rank: {}'.format(module, max_rank))

#     rdc.set_module(module)
#     rdc_s.set_module(module)
#     rdc_ssf.set_module(module)

#     # res_diff = rdc.get_res_diff_module()
#     # res_diff_dict[module] = res_diff
#     # res_diff = rdc_s.get_res_diff_module()
#     # res_diff_s_dict[module] = res_diff
#     # res_diff = rdc_ssf.get_res_diff_module()
#     # res_diff_ssf_dict[module] = res_diff
#     for neural_rank in range(1, max_rank + 1):
#         res_diff, _ = rdc_s.get_res_diff(neural_rank)
#         res_diff_s_dict[module].append(res_diff)
#         res_diff, _ = rdc_ssf.get_res_diff(neural_rank)
#         res_diff_ssf_dict[module].append(res_diff)

#         rdc.switch_time = rdc_s.switch_time # although rdc has no switch time in principle, we compute RRD using the same time window as rdc_s (same as rdc_ssf)
#         res_diff, pref_beta = rdc.get_res_diff(neural_rank)
#         res_diff_dict[module].append(res_diff)

# for module in module_list:
#     res_diff_dict[module] = np.array(res_diff_dict[module])
#     res_diff_s_dict[module] = np.array(res_diff_s_dict[module])
#     res_diff_ssf_dict[module] = np.array(res_diff_ssf_dict[module])

# # assuming all experiments have the same time length and switch time
# time = np.arange(rdc_s.tot_time)
# switch_time = rdc_s.switch_time # assuming all experiments have the same time length and switch time

# data = {'res_diff': res_diff_dict, 'res_diff_s': res_diff_s_dict, 'res_diff_ssf': res_diff_ssf_dict, 'time': time, 'switch_time': switch_time}
# hkl.dump(data, os.path.join(DATA_DIR_HOME, 'pers_res_diff.hkl'))

data = hkl.load(os.path.join(DATA_DIR_HOME, 'pers_res_diff.hkl'))
fig, axes = plt.subplots(2, 4, figsize=(10, 5))
axes = axes.flatten()
for i, module in enumerate(module_list):
    plot_traces_module(data, module, error_mode=error_mode, remove_outlier=remove_outlier, ax=axes[i], fig=fig, mean_mode=mean_mode)

    axes[i].yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    axes[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # Find offset_text to set its font size
    offset_text = axes[i].yaxis.get_offset_text()
    offset_text.set_size(10)
    offset_text.set_verticalalignment('bottom')

    if i == 0:
        axes[i].legend(fontsize=8, loc='upper right', bbox_to_anchor=(1, 1))
        axes[i].set_ylabel('Relative \n Response Difference (a.u.)')
    if i in (4, 5, 6, 7):
        axes[i].set_xlabel('Time (a.u.)')

fig.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, 'pers_res_diff_all_module.svg'))

# # plot E2 only
# fig, ax = plt.subplots(figsize=(3, 3))
# module = 'E2'
# fig, ax = plot_traces_module(data, module, error_mode=error_mode, remove_outlier=remove_outlier, ax=ax, fig=fig, mean_mode=mean_mode)
# switch_time = data['switch_time']
# ax.set_ylabel('Relative \n Response Difference (a.u.)')
# ax.set_xlabel('Time (a.u.)')
# ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1, 1))
# ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
# ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# # Find offset_text to set its font size
# offset_text = ax.yaxis.get_offset_text()
# offset_text.set_size(13)
# offset_text.set_verticalalignment('bottom')
# fig.tight_layout()
# fig.savefig(os.path.join(FIGURE_DIR, 'pers_res_diff_E2.svg'))
plt.show()
