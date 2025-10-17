# show the heatmap of example neurons
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import hickle as hkl

from border_ownership.prednet_rf_finder import compute_center_rf_mask_exp
import border_ownership.response_in_para as rip
import border_ownership.border_response_analysis as bra
from border_ownership.rf_finder import RF_Finder_Local_Sparse_Noise_Small_Memory_Center
from border_ownership.agent import Agent, Agent_RF_Wraper
from mpl_toolkits.axes_grid1 import make_axes_locatable
import border_ownership.ploter as ploter

from kitti_settings import *

output_mode_list = ['E0', 'E1', 'E2', 'E3', 'R0', 'R1', 'R2', 'R3']
neural_rank = 0 # select neuron with rank of boi, from largest (0) to smalest
bo_mean_time_init = 5; bo_mean_time_final = 19
keep_center_file = False
z_thresh = 1
smooth_sigma = 2 # smooth the z map
output_dark_path = os.path.join(DATA_DIR, 'rotating_square_dark_prednet_all_center_length50.hkl')
output_light_path = os.path.join(DATA_DIR, 'rotating_square_light_prednet_all_center_length50.hkl')
fig_format = 'pdf'
label_path = os.path.join(DATA_DIR, 'rotating_square_label_all_length50.hkl')
fig_save_path = os.path.join(FIGURE_DIR, 'example_heatmap/')
if not os.path.exists(fig_save_path): os.makedirs(fig_save_path)

for output_mode in output_mode_list:
    center_heatmap_dir = os.path.join(DATA_DIR_HOME, 'heatmap_center_{}'.format(output_mode))
    figure_file = os.path.join(fig_save_path, output_mode + '_rank{}.{}'.format(neural_rank, fig_format)) # figure save location

    # load neural responses to rotating square
    output_dark = hkl.load(output_dark_path)
    output_light = hkl.load(output_light_path)
    angle = hkl.load(label_path)['angle']
    output_center, rf_para, alpha_rf, beta_rf, gamma_rf = rip.response_in_rf_paramemterization(output_dark, output_light, angle)
    output_module = output_center[output_mode]
    output_module = output_module + 1 # avoid negative value

    boia = bra.BOI_Analyzer(output_module, alpha_rf, bo_mean_time_init=bo_mean_time_init, bo_mean_time_final=bo_mean_time_final)
    rff = RF_Finder_Local_Sparse_Noise_Small_Memory_Center()
    rff.load_center_heatmap(center_heatmap_dir)
    im = rff.query_heatmap(boia.neuron_idx[neural_rank])
    im = im + 1
    zmap = rff.query_zmap(boia.neuron_idx[neural_rank], smooth_sigma=smooth_sigma)
    fig, ax = ploter.plot_heatmap_with_contour(zmap, im, zthresh=z_thresh)
    fig.suptitle(output_mode)
    plt.tight_layout()
    fig.savefig(figure_file, format=fig_format, dpi=300)
