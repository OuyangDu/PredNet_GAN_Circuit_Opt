from border_ownership.square_part import Square_Part_Analyzer
from pprint import pprint
from border_ownership.square_part_population import remove_nan_neuron, gain_correction, plot_hist, get_mean_ci
from scipy.stats import wilcoxon, ranksums, mannwhitneyu
import matplotlib.pyplot as plt
from border_ownership.ploter import removeOutliers
from border_ownership.rf_finder import out_of_range
import numpy as np
import os
import hickle as hkl
from mpi4py import MPI
from kitti_settings import *

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
#################### Help Function ####################
def compute_p_values(arr0, arr1, keys):
    # Perform Wilcoxon signed-rank test for each column and store p-values
    p_values = np.zeros(6)

    for i in range(6):
        try:
            p_values[i] = wilcoxon(arr0[:, i], arr1[:, i], alternative="greater").pvalue
            # print(arr0[:, i], '\n', arr1[:, i], '\n', p_values[i])
        except ValueError:
            p_values[i] = 1.0
    p_values = {keys[i]: p_values[i] for i in range(6)}
    return p_values

def compute_p_values_all_condition(data_square_part_change):
    order_keys = ['CE', 'NC', 'NE', 'FC', 'FE', 'All']
    pref_with_center, nonpref_with_center, pref_no_center, nonpref_no_center = data_square_part_change[:, 0, :], data_square_part_change[:, 1, :], data_square_part_change[:, 2, :], data_square_part_change[:, 3, :]
    p_values_with_center = compute_p_values(pref_with_center, nonpref_with_center, order_keys)
    p_values_no_center = compute_p_values(pref_no_center, nonpref_no_center, order_keys)

    p_values_pref = compute_p_values(np.abs(pref_with_center), np.abs(pref_no_center), order_keys)
    p_values_nonpref = compute_p_values(np.abs(nonpref_with_center), np.abs(nonpref_no_center), order_keys)
    return p_values_with_center, p_values_no_center, p_values_pref, p_values_nonpref

#################### Hyperparameters ####################
module_list = ['E0', 'E1', 'E2', 'E3', 'R0', 'R1', 'R2', 'R3']
outmost_distance = 15

res_square_part_path = os.path.join(DATA_DIR_HOME, 'res_square_part_{}.hkl'.format(outmost_distance))
res_name = ['CE', 'NC', 'NE', 'FC', 'FE', 'all']

########## Load and draw

input_res_name = ['CE_pref', 'NC_pref', 'NE_pref', 'FC_pref', 'FE_pref', 'All_pref', 'CE_nonpref', 'NC_nonpref', 'NE_nonpref', 'FC_nonpref', 'FE_nonpref', 'All_nonpref']
target_res_name = ['All_pref', 'FE_pref', 'FC_pref', 'NE_pref', 'NC_pref', 'CE_pref', 'NC_nonpref', 'NE_nonpref', 'FC_nonpref', 'FE_nonpref', 'All_nonpref']

for module in module_list:
    data_square_part = hkl.load(res_square_part_path)[module]
    data_square_part = remove_nan_neuron(data_square_part)

    if data_square_part.shape[0] == 0: # if there is no neuron in this module
        continue

    data_square_part_change = data_square_part - data_square_part[:, :, 0, np.newaxis] # substract the first column, i.e. compute the change from the center edge. The shape of data_square_part_change is (n_neurons, n_conditions=4, edge=6). n_conditions = 0, 1, 2, 3 corresponds to (center_edge_present, preferred_side) = (T, T), (T, F), (F, T), (F, F). Edge corresponds to (CE, NC, NE, FC, FE, All)

    # compute the statistical significance
    print('Module: {}'.format(module))
    p_values_with_center, p_values_no_center, p_values_pref, p_values_non_pref = compute_p_values_all_condition(data_square_part_change)
    print('With Center')
    pprint(p_values_with_center)
    print('No Center')
    pprint(p_values_no_center)
    print('Pref')
    pprint(p_values_pref)
    print('Non Pref')
    pprint(p_values_non_pref)

    # plot the histogram
    y_mean_c, ci_lower_c, ci_upper_c = get_mean_ci(data_square_part_change, with_center=True)
    y_mean_nc, ci_lower_nc, ci_upper_nc = get_mean_ci(data_square_part_change, with_center=False)

    fig_c, ax_c = plt.subplots(figsize=(7, 3))
    plot_hist(y_mean_c, ci_lower_c, ci_upper_c, input_res_name, target_res_name, fig=fig_c, ax=ax_c, bar_color='grey')
    fig_nc, ax_nc = plt.subplots(figsize=(7, 3))
    plot_hist(y_mean_nc, ci_lower_nc, ci_upper_nc, input_res_name, target_res_name, fig=fig_nc, ax=ax_nc, bar_color='white', y_label='Response (a.u.)')

    ax_nc.spines['left'].set_visible(False)
    ax_nc.spines['right'].set_visible(True)
    ax_nc.yaxis.tick_right()
    ax_nc.yaxis.set_label_position("right")

    y_min = min(ax_c.get_ylim()[0], ax_nc.get_ylim()[0])
    y_max = max(ax_c.get_ylim()[1], ax_nc.get_ylim()[1])
    ax_c.set_ylim(y_min, y_max)
    ax_nc.set_ylim(y_min, y_max)

    ax_c.set_title('{}; Number of Units: {}'.format(module, data_square_part_change.shape[0]))

    fig_c.tight_layout()
    fig_nc.tight_layout()
    fig_c.savefig(os.path.join(FIGURE_DIR, 'square_part_change_center_{}.svg'.format(module)), format='svg')
    fig_nc.savefig(os.path.join(FIGURE_DIR, 'square_part_change_no_center_{}.svg'.format(module)), format='svg')

plt.show()
