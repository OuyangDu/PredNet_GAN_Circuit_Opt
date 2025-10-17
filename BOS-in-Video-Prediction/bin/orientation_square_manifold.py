import numpy as np
import hickle as hkl
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, MDS
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from kitti_settings import *
from border_ownership.dataset import Prednet_Ori_Square_Dataset

time_cut = [0, 10] # we havn't implemented this but 5 seems a good time cut read from the figure
output_dark_path = os.path.join(DATA_DIR, 'rotating_square_dark_prednet_all_center.hkl')
output_light_path = os.path.join(DATA_DIR, 'rotating_square_light_prednet_all_center.hkl')
label_path = os.path.join(DATA_DIR, 'rotating_square_label_all.hkl')

posd = Prednet_Ori_Square_Dataset(output_dark_path, output_light_path, label_path, output_mode='E1')
#X, Y, label_name = posd.output_data(time_cut, time_processing='slice')
X, Y, label_name = posd.output_data(time_cut, time_processing='average')

# manifold one: averaged neural responses
def scatter_color_plot(fig, n_row, n_col, latent_id, cmap, X, Y, shift=1):
    pca = PCA(n_components=2)
    res_pca = pca.fit_transform(X)

    ax_id = n_row * 100 + n_col * 10 + latent_id + shift
    ax = fig.add_subplot(ax_id)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.scatter(res_pca[:, 0], res_pca[:, 1], c=Y[:, latent_id], cmap=cmap)
    fig.colorbar(im, cax=cax, orientation='vertical', label=label_name[latent_id])

fig = plt.figure(figsize=(13.5, 3.6))
scatter_color_plot(fig, 1, 3, 0, 'hsv', X, Y)
scatter_color_plot(fig, 1, 3, 1, 'bwr', X, Y)
scatter_color_plot(fig, 1, 3, 2, 'PiYG', X, Y)
plt.tight_layout()

# Let's focus on one orientation of the edge
angle = 126
#angle = 0
angle_id = np.where(Y[:, 0] == angle)[0]
X_angle = X[angle_id]; Y_angle = Y[angle_id]

fig_angle = plt.figure(figsize=(7, 3.5))
scatter_color_plot(fig_angle, 1, 2, 1, 'bwr', X_angle, Y_angle, shift=0)
scatter_color_plot(fig_angle, 1, 2, 2, 'PiYG', X_angle, Y_angle, shift=0)
plt.tight_layout()
plt.show()
