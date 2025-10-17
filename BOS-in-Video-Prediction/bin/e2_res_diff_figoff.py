# please run res_diff_persist_figoff.py first to generate the data file
# please run quick_export_neuron_info_switch first
import border_ownership.neuron_info_processor as nip
import border_ownership.ploter as ploter
import border_ownership.persistent_ploter as pers_ploter
import copy
from brokenaxes import brokenaxes
from collections import defaultdict
import matplotlib.ticker as mticker
import border_ownership.persistent as pers
import matplotlib.pyplot as plt
import numpy as np
import os
import hickle as hkl
from kitti_settings import *

#################### Hyperparameters ####################
module_list = ['E2']
center_info_path = os.path.join(DATA_DIR_HOME, 'center_neuron_info.hkl')
error_mode = 'se'
mean_mode = 'mean'
remove_outlier = False

configs = {
    's': ('res_switch_info', 'stim_switch_info'),
    'sags': ('res_switch_ambiguous_grey_info', 'stim_switch_ambiguous_grey_info', True),
    'sgrat': ('res_switch_grating_grey_info', 'stim_switch_grating_grey_info', True),
    'spgs': ('res_switch_pixel_grey_info', 'stim_switch_pixel_grey_info')
} # The first element is the calculator's name, first value is the res_info name, second value is the stim_info name, and the third value is whether to swap gamma and beta values.

plot_keys = configs.keys()
plot_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple']
plot_labels = ['Square-Ambiguous', 'Ambiguous-Off', 'Grating-Off', 'Pixel-Off']

data_save_path = os.path.join(DATA_DIR_HOME, 'pers_figoff_res_diff.hkl')

#################### Main ####################

data = hkl.load(data_save_path)

fig_broken = plt.figure(figsize=(3, 3))
ymin = -0.005; ymax = 0.085;
ybreak0 = 0.007; ybreak1 = 0.055; ybreak2 = 0.06; ybreak3 = 0.08
bax = brokenaxes(ylims=((ymin, ybreak0), (ybreak1, ybreak2), (ybreak3, ymax)), fig=fig_broken)

fig_broken, bax = pers_ploter.plot_one_module(data, 'E2', plot_keys, plot_colors, plot_labels, error_mode, mean_mode, remove_outlier, normalize=False, fig_module=fig_broken, ax_module=bax, do_format_axis=False)

bax.set_xlabel('Relative Time (a.u.)')
bax.axs[0].set_ylabel('Relative \n Response Difference (a.u.)')
bax.legend(loc='upper right', frameon=False, fontsize=10)
bax.set_title('E2')
fig_broken.savefig(os.path.join(FIGURE_DIR, 'pers_figoff_res_diff_E2.svg'))

fig, ax = pers_ploter.plot_one_module(data, 'E2', plot_keys, plot_colors, plot_labels, error_mode, mean_mode, remove_outlier, normalize=True, sci_ytick=False)
fig.savefig(os.path.join(FIGURE_DIR, 'pers_figoff_res_diff_E2_normalize.svg'))
plt.show()
