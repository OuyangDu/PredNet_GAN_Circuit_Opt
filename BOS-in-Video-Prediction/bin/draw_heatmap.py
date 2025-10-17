from border_ownership.center_neuron_analyzer import Center_RF_Neuron_Analyzer, save_center_neuron_only
from border_ownership.rf_finder import RF_Finder_Local_Sparse_Noise
from border_ownership import ploter
import numpy as np
import border_ownership.ploter as ploter
import matplotlib.pyplot as plt
from kitti_settings import *

def get_center_square(image, square_width=50):
    center = np.array(image.shape[:2]) // 2
    half_width = square_width // 2
    square = image[
        center[0] - half_width : center[0] + half_width,
        center[1] - half_width : center[1] + half_width
    ]
    return square
# #################### Hyperparameters ####################
inspect_module = 'E1'
zthresh = 1
cmap = 'seismic'
center_neuron_rf_path = os.path.join(DATA_DIR_HOME, 'center_neuron_dict.hkl')
center_res_ori_path = os.path.join(DATA_DIR_HOME, 'square_bo_center_res_ori.npz')
# #################### Main ####################
rfn = RF_Finder_Local_Sparse_Noise()
cn_analyzer = Center_RF_Neuron_Analyzer()
cn_analyzer.load_data(center_res_ori_path, center_neuron_rf_path)
bo_info, _, _ = cn_analyzer.export_neuron_info()

heatmap = bo_info[inspect_module]['heatmap'][5]
heatmap = heatmap - 1
zmap = rfn._heatmap_to_zmap(heatmap, smooth_sigma=1, merge_black_white=False)

zmap_center = [[], []]
zmap_center[0] = get_center_square(zmap[0])
zmap_center[1] = get_center_square(zmap[1])

fig, ax = plt.subplots(1, 2, figsize=(5, 2.5))
fig, ax, cax = ploter.plot_white_black_heatmap(zmap_center, vcenter=None, cmap=cmap, cbar_label='Unit activation \n z-scored', fig=fig, ax=ax)
fig.tight_layout()

zmap = rfn._heatmap_to_zmap(heatmap, smooth_sigma=1, merge_black_white=True)
zmap_center = get_center_square(zmap)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
fig, ax, cax = ploter.single_imshow_colorbar(zmap_center, vcenter=None, cmap=cmap, cbar_label='Absolute z score', ax=ax, fig=fig)
ax.contour(zmap_center, levels=[zthresh], colors='white', linestyles='solid')
ax.axis('off')
fig.tight_layout()
plt.show()
