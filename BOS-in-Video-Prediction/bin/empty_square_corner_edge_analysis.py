# generate empty square
from kitti_settings import *
from border_ownership.square_stimuli_generator import Square_Generator
import matplotlib.pyplot as plt
from matplotlib import ticker
from border_ownership.ploter import add_ax_contour
import border_ownership.border_response_analysis as bra
from border_ownership.prednet_res_to_stimuli import get_neural_response_to_para
from border_ownership.empty_square import plot_neural_res
import hickle as hkl
import numpy as np

output_mode_list = ['E0', 'E1', 'E2', 'E3', 'R0', 'R1', 'R2', 'R3']
#output_mode_list = ['R2']
neural_rank = 0
width, height = 160, 128
dark_grey = 255 // 3
light_grey = 255 * 2 // 3
length_square = 50
strip_id_list = [[None], [1, 7], [0]]
strip_id_name = ['Full', 'No_Corner', 'No_Edge_Center']
pixel_format = '1' # pixel value ranges from 0 to 1
n_frames = 20

fig_path_dir = os.path.join(FIGURE_DIR, 'empty_square/')
is_exist = os.path.exists(fig_path_dir)
if not is_exist: os.makedirs(fig_path_dir)

# rotating files, this is used for find neural id by rank
output_dark_path = os.path.join(DATA_DIR, 'rotating_square_dark_prednet_all_center_length50.hkl')
output_light_path = os.path.join(DATA_DIR, 'rotating_square_light_prednet_all_center_length50.hkl')
label_path = os.path.join(DATA_DIR, 'rotating_square_label_all_length50.hkl')
#################### hyperparameters finished
for output_mode in output_mode_list:
    center_heatmap_dir = os.path.join(DATA_DIR_HOME, 'heatmap_center_{}'.format(output_mode)) # for finding the RF

    # get target neural id and angle
    neural_id, edge_ori = bra.get_neural_id_by_rank(output_dark_path, output_light_path, label_path, output_mode, neural_rank)

    # look its neural response to empty square

    pos_neural_res, edge_ori, para_list, pos_video_batch = get_neural_response_to_para(edge_ori, light_grey, None, True, strip_id_list, n_frames, pixel_format, output_mode=output_mode, mode='strip', verbose=True)
    neg_neural_res, edge_ori, para_list, neg_video_batch = get_neural_response_to_para(edge_ori, light_grey, None, False, strip_id_list, n_frames, pixel_format, output_mode=output_mode, mode='strip', verbose=True)
    pos_neural_res += 1
    neg_neural_res += 1
    pos_neural_res = pos_neural_res[..., neural_id]
    neg_neural_res = neg_neural_res[..., neural_id]

    for i in range(len(strip_id_list)):
        pos_image = pos_video_batch[i][0]
        neg_image = neg_video_batch[i][0]
        pos_single_neural_res = pos_neural_res[i]
        neg_single_neural_res = neg_neural_res[i]
        strip_name = strip_id_name[i]
        fig, ax = plot_neural_res(pos_image, neg_image, pos_single_neural_res, neg_single_neural_res, strip_name, center_heatmap_dir, neural_id, n_frames=n_frames)
        fig.suptitle(output_mode + ' ' + strip_name)
        fig.tight_layout()
        fig_path = os.path.join(fig_path_dir, 'empty_square_res_{}_rank{}_{}.pdf'.format(output_mode, neural_rank, strip_name))
        fig.savefig(fig_path, format='pdf', dpi=300)

    # average response bias
    pos_neural_res_mean = np.mean(pos_neural_res, axis=1)
    neg_neural_res_mean = np.mean(neg_neural_res, axis=1)
    response_bias = np.abs(pos_neural_res_mean - neg_neural_res_mean) / (pos_neural_res_mean + neg_neural_res_mean) * 2
    x = range(len(strip_id_name))
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot()
    ax.scatter(x, response_bias)
    ax.set_xticks(x)
    ax.set_xticklabels(labels=strip_id_name, fontsize='x-small')
    ax.set_ylabel('BOI')
    fig_path = os.path.join(fig_path_dir, 'empty_square_res_bias_{}_rank{}.pdf'.format(output_mode, neural_rank))
    fig.tight_layout()
    fig.suptitle(output_mode + ' empty square')
    fig.savefig(fig_path)
