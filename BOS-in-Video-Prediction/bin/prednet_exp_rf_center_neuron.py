import os
import numpy as np
import matplotlib.pyplot as plt

from border_ownership.rf_finder import RF_Finder_Local_Sparse_Noise_Small_Memory_Center
from border_ownership.agent import Agent, Agent_RF_Wraper
from mpl_toolkits.axes_grid1 import make_axes_locatable

from kitti_settings import *

output_mode = 'E2'
input_shape = [128, 160, 3]
query_neural_id_list = np.arange(0, 100, 10) # E2
#query_neural_id_list = np.arange(0, 3) # A0
heatmap_dir = os.path.join(DATA_DIR_HOME, 'heatmap_{}'.format(output_mode))
center_heatmap_dir = os.path.join(DATA_DIR_HOME, 'heatmap_center_{}'.format(output_mode))
generate_center_file = False
z_thresh = 1

rff = RF_Finder_Local_Sparse_Noise_Small_Memory_Center(heatmap_dir, input_shape=input_shape, z_thresh=z_thresh)
if generate_center_file: rff.keep_center_heatmap(center_heatmap_dir)
rff.load_center_heatmap(center_heatmap_dir)

for qid in query_neural_id_list:
    heatmap = rff.query_heatmap(qid)
    fig, ax = plt.subplots(1, 3, figsize=(8, 2.))

    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax[0].imshow(heatmap[0], cmap='seismic')
    ax[0].set_title('Black Heatmap')
    fig.colorbar(im, cax=cax, orientation='vertical', label='neural response')

    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax[1].imshow(heatmap[1], cmap='seismic')
    ax[1].set_title('White Heatmap')
    fig.colorbar(im, cax=cax, orientation='vertical', label='neural response')

    rf = rff.query_rf(qid)
    im = ax[2].imshow(rf, cmap='binary')
    ax[2].set_title('RF')

    plt.tight_layout()
    plt.show()
