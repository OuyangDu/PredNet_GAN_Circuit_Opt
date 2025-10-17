import os
from kitti_settings import *
import numpy as np
import border_ownership.border_response_analysis as bra
from border_ownership.prednet_res_to_stimuli import get_neural_response_to_para
import border_ownership.empty_square as esquare
#plot_neural_res, Empty_Square, empty_square_part_image, compute_empty_square_part_neural_res, plot_empty_square_part_image
from border_ownership.ploter import add_ax_contour
import matplotlib.pyplot as plt

#output_mode_list = ['E0', 'E1', 'E2', 'E3', 'R0', 'R1', 'R2', 'R3']
output_mode_list = ['E2', 'E1']
neural_rank = 0
time_window = [5, 20]
length_square = 50
# rotating files, this is used for find neural id by rank
output_dark_path = os.path.join(DATA_DIR, 'rotating_square_dark_prednet_all_center_length{}.hkl'.format(length_square))
output_light_path = os.path.join(DATA_DIR, 'rotating_square_light_prednet_all_center_length{}.hkl'.format(length_square))
label_path = os.path.join(DATA_DIR, 'rotating_square_label_all_length{}.hkl'.format(length_square))

for output_mode in output_mode_list:
    center_heatmap_dir = os.path.join(DATA_DIR_HOME, 'heatmap_center_{}'.format(output_mode)) # for finding the RF
    fig_save_dir = os.path.join(FIGURE_DIR, 'empty_square_part/')
    if not os.path.exists(fig_save_dir): os.mkdir(fig_save_dir)

    # compute the neural response with only keep 0
    neural_id, edge_ori = bra.get_neural_id_by_rank(output_dark_path, output_light_path, label_path, output_mode, neural_rank, time_window[0], time_window[1])

    config_list = [[True, True], [True, False], [False, True], [False, False]] # first element is with center, second is beta
    config_name = ['wc: {}; b: {}'.format(str(wc)[0], str(beta)[0]) for wc, beta in config_list]
    #es = esquare.Empty_Square(edge_ori=edge_ori, length_square=length_square)
    es = esquare.Empty_Square(edge_ori=edge_ori, length_square=length_square)

    #################### Show input images
    def plot_image_center_beta(with_center, beta):
        images_wt_bt = esquare.empty_square_part_image(es, with_center=with_center, beta=beta)
        fig, ax = esquare.plot_empty_square_part_image(images_wt_bt, beta=beta, with_rf_contour=True, center_heatmap_dir=center_heatmap_dir, neural_id=neural_id)
        fig.suptitle(output_mode)
        file_path = os.path.join(fig_save_dir, 'image_w{}_b{}_{}.pdf'.format(with_center, beta, output_mode))
        fig.savefig(file_path, format='pdf')
        return None
    plot_image_center_beta(True, True)
    plot_image_center_beta(True, False)

    ##################### Neural response
    index = np.arange(8)
    neural_res_change = []
    for with_center, beta in config_list:
        neural_res_change_temp = esquare.compute_empty_square_part_neural_res(es, with_center=with_center, beta=beta, output_mode=output_mode, neural_id=neural_id)
        neural_res_change.append(neural_res_change_temp)

    fig, ax = plt.subplots(figsize=(3, 3))
    for i, nrc in enumerate(neural_res_change):
        ax.plot(index, nrc, label=config_name[i])
        ax.scatter(index, nrc)
        ax.set_xlabel('empty square part id')
        ax.set_ylabel('change of neural response')
    fig.suptitle(output_mode)
    fig.tight_layout()
    file_path = os.path.join(fig_save_dir, 'neural_response_change_id{}_module{}.pdf'.format(neural_id, output_mode))
    plt.legend()
    fig.savefig(file_path, format='pdf')

    fig, ax = esquare.plot_sequential_res_on_square(neural_res_change, config_list)
    fig.suptitle(output_mode)
    file_path = os.path.join(fig_save_dir, 'mat_neural_response_change_id{}_module{}.pdf'.format(neural_id, output_mode))
    fig.savefig(file_path, format='pdf')
#plt.show()
