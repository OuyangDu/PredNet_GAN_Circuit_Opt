# compare the res difference between persistance (square-ambiguous) and fig-off (ambiguous-grey)
# please run quick_export_neuron_info_switch first
import border_ownership.neuron_info_processor as nip
import border_ownership.ploter as ploter
import border_ownership.persistent_ploter as pers_ploter
import copy
from collections import defaultdict
import matplotlib.ticker as mticker
import border_ownership.persistent as pers
import matplotlib.pyplot as plt
import numpy as np
import os
import hickle as hkl
from kitti_settings import *

#################### Hyperparameters ####################
module_list = ['R0', 'R1', 'R2', 'R3', 'E0', 'E1', 'E2', 'E3']
center_info_path = os.path.join(DATA_DIR_HOME, 'center_neuron_info.hkl')
error_mode = 'se'
mean_mode = 'mean'
remove_outlier = False

configs = {
    's': ('res_switch_info', 'stim_switch_info'),
    'sags': ('res_switch_ambiguous_grey_info', 'stim_switch_ambiguous_grey_info', True),
    'sgrat': ('res_switch_grating_grey_info', 'stim_switch_grating_grey_info', True),
    'spgs': ('res_switch_pixel_grey_info', 'stim_switch_pixel_grey_info')
} # key value is the calculator's name, first value is the res_info name, second value is the stim_info name, and the third value is whether to swap gamma and beta values.

plot_keys = configs.keys()
plot_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple']
# plot_labels = ['Square-Ambiguous', 'Ambiguous-Off', 'Grating-Off', 'Grey-Off']
plot_labels = ['Square-Ambiguous', 'Ambiguous-Off', 'Grating-Off', 'Pixel-Off']

data_save_path = os.path.join(DATA_DIR_HOME, 'pers_figoff_res_diff.hkl')

#################### Main ####################
# load data
data = hkl.load(center_info_path)
bo_info, unique_orientation = data['bo_info'], data['unique_orientation']
# create calculators for computing response difference
rdc_calculators = pers.configure_calculators(data, bo_info, unique_orientation, configs)
# get time and switch time
time = np.arange(rdc_calculators['s'].tot_time)
switch_time = rdc_calculators['s'].switch_time # assuming all experiments have the same time length and switch time

# Dictionary to hold results for each configuration
res_diff_dict = defaultdict(dict)

# for config_key in rdc_calculators:
#     for module in module_list:
#         print('Processing {} {}'.format(config_key, module))
#         rdc_calculators[config_key].set_module(module)
#         res_diff_module = rdc_calculators[config_key].get_res_diff_module()
#         res_diff_dict[config_key][module] = np.abs(res_diff_module)

# res_diff_dict['time'] = time
# res_diff_dict['switch_time'] = switch_time

# hkl.dump(res_diff_dict, data_save_path)

data = hkl.load(data_save_path)
fig, axes = pers_ploter.plot_all_modules(data, module_list, plot_keys, plot_colors, plot_labels, error_mode, mean_mode, remove_outlier, normalize=True, ylabel= 'Relative and Rescaled \n Response Difference (a.u.)')
fig.savefig(os.path.join(FIGURE_DIR, 'pers_figoff_res_diff_all_module_norm.svg'))
plt.show()
