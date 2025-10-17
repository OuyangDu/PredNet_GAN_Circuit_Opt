import os
import pprint
import hickle as hkl
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import linregress
from border_ownership.ablation_visu import get_mse
from border_ownership.ploter import plot_layer_boxplot_helper, plot_layer_violin_helper
from border_ownership.ablation_stat import Ablation_Data_Reader, avg_video_num_unit_boot, compute_pdf
from border_ownership.ploter import error_bar_plot
# from sklearn.linear_model import LinearRegression
from kitti_settings import *

####################
# Help functions
####################
def reshape_data(avg_data):
    # reshape to dataset format, so the n_units shape is [n_units * n_video, 1], True_rpmse and False_rpmse shape is [n_units * n_video, 1]
    n_video = avg_data[module]['True_rpmse'].shape[1] # repeat n_units for True_rpmse and False_rpmse
    x = np.repeat(avg_data[module]['n_units'][:, None], n_video, axis=1).reshape([-1, 1]);
    y_t = avg_data[module]['True_rpmse'].reshape([-1, 1]);
    y_f = avg_data[module]['False_rpmse'].reshape([-1, 1])
    return x, y_t, y_f

def compute_slope(x, y_t, y_f):
    id_t_s = np.random.choice(x.shape[0], x.shape[0], replace=True)
    x_t_s = x[id_t_s]; y_t_s = y_t[id_t_s];
    k_t_s = np.linalg.lstsq(x_t_s, y_t_s, rcond=None)[0][0]

    id_f_s = np.random.choice(x.shape[0], x.shape[0], replace=True)
    x_f_s = x[id_f_s]; y_f_s = y_f[id_f_s];
    k_f_s = np.linalg.lstsq(x_f_s, y_f_s, rcond=None)[0][0]

    k_diff_s = k_t_s - k_f_s
    return k_t_s, k_f_s, k_diff_s

def calculate_pvalue(t, scalar):
    quantile_position = np.mean(np.array(t) <= scalar)
    pvalue = 2 * min(quantile_position, 1 - quantile_position)
    return pvalue

def plot_module(avg_data, k_t, k_f, module, fig, ax):
    x = avg_data[module]['n_units']
    y_t = avg_data[module]['True_rpmse']
    y_f = avg_data[module]['False_rpmse']

    error_bar_plot(x, y_t, fig=fig, ax=ax, color='tab:blue', label='BO', error_mode='se', mean_mode='mean', with_line=False, error_band=False, with_scatter=True)
    error_bar_plot(x, y_f, fig=fig, ax=ax, color='tab:orange', label='Non-BO', error_mode='se', mean_mode='mean', with_line=False, error_band=False, with_scatter=True)

    # draw the line
    y_t_line = x * np.mean(k_t[module])
    y_f_line = x * np.mean(k_f[module])
    ax.plot(x, y_t_line, color='tab:blue', label='BO')
    ax.plot(x, y_f_line, color='tab:orange', label='Non-BO')

    # draw the ci of the line
    y_t_ci_low = x * np.percentile(k_t[module], 2.5)
    y_f_ci_low = x * np.percentile(k_f[module], 2.5)
    y_t_ci_up = x * np.percentile(k_t[module], 97.5)
    y_f_ci_up = x * np.percentile(k_f[module], 97.5)
    ax.fill_between(x.flatten(), y_t_ci_low.flatten(), y_t_ci_up.flatten(), color='tab:blue', alpha=0.1)
    ax.fill_between(x.flatten(), y_f_ci_low.flatten(), y_f_ci_up.flatten(), color='tab:orange', alpha=0.1)

def set_axis(ax, position, title=None, sign=None):
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    if position[1] == 0 and position[0] == 0:
        ax.set_ylabel('Relative Prediction MSE')
        ax.legend(fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.text(0.5, 1.2, sign, ha='center', va='bottom', color='red', transform=ax.transAxes, fontsize=16)

def sign_symbol(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'
########################################
# Hyperparameters
########################################
n_boot = 10000
module_list = ['E0', 'E1', 'E2', 'E3', 'R0']
random_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # random_id for random initialization
video_type = 'trans'
adr = Ablation_Data_Reader(video_type, random_id) # square, random, trans, kitti
avg_data = {}

####################
# generate data
####################
rpmse_diff = {}
for module in module_list:
    avg_data_temp = avg_video_num_unit_boot(adr, module, shuffle_is_bo=False, avg_video=False) # average across video_resample_id and unit_resample_id. avg_data_temp should have three keys, n_units is the number of units ablated, True_rpmse is the relative prediction MSE for BO ablation, False_rpmse is the relative prediction MSE for non-BO ablation. The shape of n_units, True_rpmse and False_rpmse are [n_units]
    avg_data[module] = avg_data_temp

####################
# compute k_diff distribution and p-value
####################

p_value = {}
k_diff = {}; k_t = {}; k_f = {}
for module in module_list:
    k_diff[module] = []; k_t[module] = []; k_f[module] = []
    x, y_t, y_f = reshape_data(avg_data)

    for _ in range(n_boot):
        k_t_s, k_f_s, k_diff_s = compute_slope(x, y_t, y_f)

        k_t[module].append(k_t_s)
        k_f[module].append(k_f_s)
        k_diff[module].append(k_diff_s)

    p_value[module] = calculate_pvalue(k_diff[module], 0)

pp = pprint.PrettyPrinter(indent=4)
print(f'{video_type} p-value:')
pp.pprint(p_value)

####################
# plot
####################
e_module_list = ['E0', 'E1', 'E2', 'E3']
r_module_list = ['R0']
fig, ax = plt.subplots(2, 4, figsize=(10, 5))
for i, module in enumerate(e_module_list):
    plot_module(avg_data, k_t, k_f, module, fig, ax[0, i])
    set_axis(ax[0, i], (0, i), title=module, sign=sign_symbol(p_value[module]))

for i, module in enumerate(r_module_list):
    plot_module(avg_data, k_t, k_f, module, fig, ax[1, i])
    set_axis(ax[1, i], (1, i), title=module, sign=sign_symbol(p_value[module]))
    ax[1, i].set_xlabel('Number of Units')

fig.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, f'{video_type}_pred_diff_slope.svg'))
plt.show()
