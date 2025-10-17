# please generate center heatmap data from prednet_rf.py
import os
import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl

from border_ownership.prednet_rf_finder import compute_center_rf_mask_exp
import border_ownership.response_in_para as rip
import border_ownership.border_response_analysis as bra
from border_ownership.rf_finder import RF_Finder_Local_Sparse_Noise_Small_Memory_Center
from border_ownership.agent import Agent, Agent_RF_Wraper
from mpl_toolkits.axes_grid1 import make_axes_locatable
import border_ownership.ploter as ploter

from kitti_settings import *

#output_mode_list = ['E0', 'E1', 'E2', 'E3', 'R0', 'R1', 'R2', 'R3']
output_mode_list = ['E2', 'R1']
neural_rank = 0 # select neuron with rank of boi, from largest (0) to smalest
bo_mean_time_init = 5; bo_mean_time_final = 19
z_thresh = 1
length_square = 50
contour_smooth_sigma = 2 # smooth the z map
output_dark_path = os.path.join(DATA_DIR, 'rotating_square_dark_prednet_all_center_length{}.hkl'.format(length_square))
output_light_path = os.path.join(DATA_DIR, 'rotating_square_light_prednet_all_center_length{}.hkl'.format(length_square))
fig_format = 'pdf'
label_path = os.path.join(DATA_DIR, 'rotating_square_label_all_length{}.hkl'.format(length_square))
fig_save_path = os.path.join(FIGURE_DIR, 'example_res/')
if not os.path.exists(fig_save_path): os.makedirs(fig_save_path)

#data = {}
for output_mode in output_mode_list:
    center_heatmap_dir = os.path.join(DATA_DIR_HOME, 'heatmap_center_{}'.format(output_mode)) # for finding the RF
    response_figure_file = os.path.join(fig_save_path, output_mode + '_rank{}_z{}.{}'.format(neural_rank, z_thresh, fig_format)) # figure save location
    orientation_figure_file = os.path.join(fig_save_path, output_mode + '_rank{}_ori.{}'.format(neural_rank, fig_format)) # figure save location

    # load neural responses to rotating square
    output_dark = hkl.load(output_dark_path)
    output_light = hkl.load(output_light_path)
    angle = hkl.load(label_path)['angle']
    output_center, rf_para, alpha_rf, beta_rf, gamma_rf = rip.response_in_rf_paramemterization(output_dark, output_light, angle)
    output_module = output_center[output_mode]
    output_module = output_module + 1 # avoid negative value

    # load analyzer
    boia = bra.BOI_Analyzer(output_module, alpha_rf, bo_mean_time_init=bo_mean_time_init, bo_mean_time_final=bo_mean_time_final)

    ######### save response data
    #data[output_mode + '_response'] = output_module[..., boia.neuron_idx[neural_rank]]
    #data[output_mode + '_ori_para'] = alpha_rf
    #data[output_mode + '_beta_para'] = beta_rf
    #data[output_mode + '_gamma_para'] = beta_rf


    # plot example neural responses with RF
    fig, stim_ax_list, ax_res = ploter.plot_example_neuron_response_and_input(output_center['X'], output_module, boia.alpha_idx[neural_rank], boia.neuron_idx[neural_rank], rf_mask=None, figsize=(8, 3.8)) # plot input stimuli and neural responses

    rff = RF_Finder_Local_Sparse_Noise_Small_Memory_Center() # add rf contour
    rff.load_center_heatmap(center_heatmap_dir)
    zmap = rff.query_zmap(boia.neuron_idx[neural_rank], smooth_sigma=contour_smooth_sigma, merge_black_white=True)
    [ax.contour(zmap, levels=[z_thresh], colors='black', linestyles=[(0, (1,1))] ) for ax in stim_ax_list]

    fig.suptitle(output_mode)
    fig.savefig(response_figure_file, format=fig_format, dpi=300)

    # plot boi at different orientataion
    angle, boi = boia.neural_boi_orientation(neural_rank)

    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(projection='polar')
    ax.scatter(np.deg2rad(angle), boi)
    ax.vlines(np.deg2rad(angle), 0, boi)
    ax.set_title(output_mode)
    fig.tight_layout()
    fig.savefig(orientation_figure_file)

######### save response data
#from scipy.io import savemat
#savemat('neural_response_ori.mat', data)

plt.show()
