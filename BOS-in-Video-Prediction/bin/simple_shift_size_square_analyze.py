# please generate center heatmap data from prednet_rf.py
import os
import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl

from border_ownership.prednet_rf_finder import compute_center_rf_mask_exp
import border_ownership.response_in_para as rip
import border_ownership.border_response_analysis as bra
from border_ownership.rf_finder import RF_Finder_Local_Sparse_Noise_Small_Memory_Center
from border_ownership.prednet_res_to_stimuli import get_neural_response_to_para
from mpl_toolkits.axes_grid1 import make_axes_locatable
import border_ownership.ploter as ploter

from kitti_settings import *

#output_mode_list = ['E0', 'E1', 'E2', 'E3', 'R0', 'R1', 'R2', 'R3']
output_mode_list = ['E2']
neural_rank = 0 # select neuron with rank of boi, from largest (0) to smalest
bo_mean_time_init = 5; bo_mean_time_final = 20
mode = 'size'

# square parameters
width, height = 160, 128
dark_grey = 255 // 3
light_grey = 255 * 2 // 3
length_square = 50
n_frames = 20
pixel_format = '1'
shift_dis_list = np.arange(-30, 30, 3)
size_list = np.arange(10, 100, 10)

gamma = False

if mode == 'shift': para_list = shift_dis_list
elif mode == 'size': para_list = size_list

###### trained PredNet
#rotating files, this is used for find neural id by rank
weights_file, json_file = None, None
output_dark_path = os.path.join(DATA_DIR, 'rotating_square_dark_prednet_all_center_length50.hkl')
output_light_path = os.path.join(DATA_DIR, 'rotating_square_light_prednet_all_center_length50.hkl')
label_path = os.path.join(DATA_DIR, 'rotating_square_label_all_length50.hkl')
######## untrained PredNet
#weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/' 'prednet_kitti_weights_untrain_small.hdf5')  # where weights will be saved
#json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model_untrain_small.json')
#length_square = 50
#file_name_tail = 'untrain_small'
#output_dark_path = os.path.join(DATA_DIR, 'rotating_square_dark_prednet_all_center_length{}_{}.hkl'.format(length_square, file_name_tail))
#output_light_path = os.path.join(DATA_DIR, 'rotating_square_light_prednet_all_center_length{}_{}.hkl'.format(length_square, file_name_tail))
#label_path = os.path.join(DATA_DIR, 'rotating_square_label_all_length{}_{}.hkl'.format(length_square, file_name_tail))

# figure save path
fig_format = 'pdf'
fig_save_path = os.path.join(FIGURE_DIR, 'shift_size_res/')
if not os.path.exists(fig_save_path): os.makedirs(fig_save_path)

c_light = 'tab:blue' # response curve figure
c_dark = 'tab:green'

def plot_para_response(neural_res_dark, neural_res_light, para):
    fig = plt.figure(figsize=(3, 3))
    ax_para = fig.add_subplot()
    ax_para.scatter(para, neural_res_dark, color=c_dark)
    ax_para.plot(para, neural_res_dark, color=c_dark)
    ax_para.scatter(para, neural_res_light, color=c_light)
    ax_para.plot(para, neural_res_light, color=c_light)
    ax_para.set_xlabel('{} (unit is pixel)'.format(mode))
    ax_para.set_ylabel('Firing rate')

    fig.tight_layout()
    return fig, ax_para

for output_mode in output_mode_list:
    # get the neural id and preferred orientation depending on the output mode and neural rank
    neural_id, edge_ori = bra.get_neural_id_by_rank(output_dark_path, output_light_path, label_path, output_mode, neural_rank)
    # neural responses to different para
    neural_res_dark, edge_ori, para_list = get_neural_response_to_para(edge_ori, light_grey, dark_grey, beta=(not gamma), para_list=para_list, n_frames=n_frames, pixel_format=pixel_format, output_mode=output_mode, mode=mode, prednet_json_file=json_file, prednet_weights_file=weights_file)
    neural_res_light, edge_ori, para_list = get_neural_response_to_para(edge_ori, dark_grey, light_grey, beta=gamma, para_list=para_list, n_frames=n_frames, pixel_format=pixel_format, output_mode=output_mode, mode=mode, prednet_json_file=json_file, prednet_weights_file=weights_file)
    # get the target neural response, and take the average
    neural_res_dark = neural_res_dark[..., neural_id]
    neural_res_dark += 1
    neural_res_dark = np.mean(neural_res_dark[:, bo_mean_time_init:bo_mean_time_final], axis=1)
    neural_res_light = neural_res_light[..., neural_id]
    neural_res_light += 1
    neural_res_light = np.mean(neural_res_light[:, bo_mean_time_init:bo_mean_time_final], axis=1)
    
    fig, ax = plot_para_response(neural_res_dark, neural_res_light, para_list)

    ax.set_title(output_mode)
    fig.tight_layout()
    para_figure_file = os.path.join(fig_save_path, output_mode + '_rank{}_{}.{}'.format(neural_rank, mode, fig_format)) # figure save location
    fig.savefig(para_figure_file)
    plt.show()
